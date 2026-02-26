from __init__ import *
from cnn import *
import initialsetting as ins
reload(ins)
import imaging as IM
reload(IM)
import training as TR
reload(TR)
# from zipfile import ZipFile, ZIP_DEFLATED


class TestStockDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.stock_folders = os.listdir(self.root_dir)
        self.samples = self._load_samples()

    def _load_samples(self):
        samples = []
        for stock_dir in self.stock_folders:
            stock_path = os.path.join(self.root_dir, stock_dir)
            for pkl_file in os.listdir(stock_path):
                if pkl_file.endswith('.pkl'):
                    pkl_path = os.path.join(stock_path, pkl_file)
                    with open(pkl_path, 'rb') as p:
                        data = pickle.load(p)
                        for timestamp in data.keys():
                            samples.append((pkl_path, timestamp))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        pkl_path, timestamp = self.samples[idx]
        didx, tidx = timestamp.split(' ')[0], timestamp.split(' ')[-1]
        match = re.search(r'(\d{6})([A-Z]{2})/', pkl_path)
        with open(pkl_path, 'rb') as p:
            data = pickle.load(p)
        image, ret_long, ret_short = data[timestamp]
        return image, ret_long, ret_short, match.group(1), didx, tidx  
        # match.group(1) refers to the stock code


class Testing(ins.InitialSetting, CNN49u, CNN9u):
    def __init__(self, yaml_dir=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'config.yaml'), data_source='Test', GPU_label='0'):
        ins.InitialSetting.__init__(self, yaml_dir=yaml_dir)
        CNN49u.__init__(self)
        CNN9u.__init__(self)

        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.batch_size = self.config.TRAIN.BATCH_SIZE
        os.environ["CUDA_LAUNCH_BLOCKING"] = GPU_label
        # os.environ["CUDA_VISIBLE_DEVICES"] = GPU_label

        self.project_name = 'Train_{}_{}-Infer_{}_{}'.format(
            self.config.TRAIN.START_DATE, 
            self.config.TRAIN.END_DATE, 
            self.config.INFERENCE.START_DATE, 
            self.config.INFERENCE.END_DATE
        )

        self.edition = self.config.TRAIN.EDITION

        # Condition the shape of the output
        self.commoncache_dir = self.config.PATHS.COMMONCACHE_DIR
        self.tradedays_dir = os.path.join(self.commoncache_dir, '__universe/dates.NI')
        self.uid_dir = os.path.join(self.commoncache_dir, '__universe/uid.N128C')

        self.no_interval = int(re.search(r'\d+', self.config.MODEL).group())
        self.n_shift = self.config.TRAIN.PREDICT_WIN + 1

        self.tradeday_all = np.memmap(
            self.tradedays_dir, 
            mode='r', 
            dtype=self.dtype_map(self.tradedays_dir[-1])
        )
        if data_source == 'Test':
            self.root_dir = os.path.join(self.config.PATHS.IMAGE_DATA_DIR, self.project_name, 'Infer', 'PixelData')
            self.test_start, self.test_end = self.config.INFERENCE.START_DATE, self.config.INFERENCE.END_DATE
        elif data_source == 'Train_all':
            self.root_dir = os.path.join(self.config.PATHS.IMAGE_DATA_DIR, self.project_name, 'Train_all', 'PixelData')
            self.test_start, self.test_end = self.config.TRAIN.START_DATE, self.config.TRAIN.END_DATE

        if self.no_interval == 49:
            # TIDX
            self.intervals = []
            for minute in np.arange(0, 125, 5):
                hour = 9 + (minute + 30) // 60
                minute_ = (minute + 30) % 60
                self.intervals.append(f'{hour:02d}:{minute_:02d}')
            for minute in np.arange(5, 125, 5):
                hour = 13 + minute // 60
                minute_ = minute % 60
                self.intervals.append(f'{hour:02d}:{minute_:02d}')
        else:
            pass

        day_mask = (self.tradeday_all >= self.test_start) & (self.tradeday_all <= self.test_end)
        day_mask = np.roll(day_mask, 1)
        day_mask[0] = False
        self.tradeday = self.tradeday_all[day_mask]
        self.n_tradeday = len(self.tradeday)  # Number of Trade days

        self.uids = np.memmap(
            self.uid_dir, 
            mode='r', 
            dtype=self.dtype_map(self.uid_dir[-4:])
        )
        self.n_stocks = len(self.uids)

        self.cache_shape = (len(self.tradeday_all), self.n_stocks, len(self.intervals))

        self.model = CNN49u() if self.config.MODEL == 'CNN49u' else CNN9u()
        self.model.to(self.device)
        state_dict = torch.load(self.config.TRAIN.MODEL_SAVE_FILE)
        self.model.load_state_dict(state_dict['model_state_dict'])
        self.criterion = nn.BCELoss().to(self.device)

        start_time = time.time()
        self.dataset = TestStockDataset(self.root_dir)
        end_time = time.time()
        elapsed = end_time - start_time
        print(f'Data Retrieval Complete! | Time Use: {elapsed:.3f} second(s).')

    def dtype_map(self, end):
        mappings = {
            'C':'str',
            'I':'int32',
            'L':'int64',
            'c':'bool',
            'd':'float64',
            'f':'float32',
            'i':'uint32',
            '128C':'U32',
        }
        return mappings[end]

    def generate_cache(self, logits, save_dir):
        logits_32 = logits.astype('float32')
        logits_32.tofile(
            save_dir + 'CNN.delay{}.N{}.{}f'.format(
              self.config.TRAIN.PREDICT_WIN + 1,
              self.no_interval,
              self.n_stocks
              )
        )
        print('Cache Saved')

    def cnn_testing(self, no_partition=4, testing_all=True):
        if testing_all:
            print(f'Testing Period: {self.test_start} - {self.test_end}')
            test_loss_stat, test_acc, logits = self.testing_all()
            mapping = {
                'DIDX': self.tradeday,
                'Ticker': self.uids,
                'TIDX': np.array(self.intervals)
            }
        else:
            test_loss_stat, test_acc_stat, test_logits = [], [], []
            partitions = self.partition(
                start=self.test_start, 
                end=self.test_end, 
                no_partition=no_partition
            )
            for test_idx in range(no_partition):
                print(f'Testing subperiod: {partitions[test_idx]} - {partitions[test_idx + 1]}')
                test_loss, test_acc, logits = self.testing_each()
                test_loss_stat.append(test_loss)
                test_acc_stat.append(test_acc)
                test_logits.append(logits)

        # Save test_loss_stat, test_acc, logits, mapping
        if 'Infer' in self.root_dir:
            outcome_dir = os.path.join(self.config.PATHS.PROJECT_DIR, 'TestOutcome', self.project_name, self.edition) + '/'
        elif 'Train_all' in self.root_dir:
            outcome_dir = os.path.join(self.config.PATHS.PROJECT_DIR, 'TrainAllOutcome', self.project_name, self.edition) + '/'
        os.makedirs(outcome_dir, exist_ok=True)
        with open(outcome_dir + 'test_logits.pkl', 'wb') as p:
            pickle.dump(logits, p)
        print('Logits for Test: Saved!')

        # Generate Cache File
        self.generate_cache(logits, outcome_dir)

        with open(outcome_dir + 'indexing.pkl', 'wb') as p:
            pickle.dump(mapping, p)
        print('Mapping for Test: Saved!')

        with open(outcome_dir + 'test_stats.pkl', 'wb') as p:
            pickle.dump({'Loss': test_loss_stat, 'Accuracy': float(test_acc)}, p)
        print('Test Stats: Saved!')

    def time_shift(self, didx, tidx):
        tidx_target_idx = self.intervals.index(
            tidx if not tidx.startswith('9:') else tidx.zfill(5)
        )
        time_idx = (tidx_target_idx + self.n_shift) % self.no_interval
        date_idx = np.where(
            self.tradeday_all > int(didx.replace('-', ''))
        )[0][(tidx_target_idx + self.n_shift) // self.no_interval - 1]
        return date_idx, time_idx

    def testing_all(self):
        self.model.eval()
        test_loss, test_acc = 0, 0
        test_logits = np.full(self.cache_shape, np.nan)
        # test_logits = np.full((self.n_tradeday, self.n_stocks, self.no_interval), np.nan)

        test_loader = torch.utils.data.DataLoader(
            dataset=self.dataset, 
            batch_size=self.batch_size, 
            shuffle=False
        )
        notification = [len(test_loader) * i // 10 for i in range(1, 11)]  # 10 Notification Points

        for batch_idx, (data, ret_short, ret_long, ticker, didx, tidx) in enumerate(test_loader):
            # Record Time Usage
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()

            if batch_idx + 1 == len(test_loader):
                batch_size = data.size(0)
            else:
                batch_size = self.batch_size

            if self.config.TRAIN.LABEL == 'RET_SHORT':
                target = ret_short
            elif self.config.TRAIN.LABEL == 'RET_LONG':
                target = ret_long

            target = (1 - target).unsqueeze(1) @ torch.LongTensor([[1., 0.]]).unsqueeze(1).T + \
                     target.unsqueeze(1) @ torch.LongTensor([[0, 1.]]).unsqueeze(1).T
            target = target.to(torch.float32)

            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            test_loss += loss.item() * data.size(0)
            acc = (output.argmax(1) == target.argmax(1)).sum()
            test_acc += acc

            for i in range(batch_size):
                date_idx, time_idx = self.time_shift(didx[i], tidx[i])
                stock_idx = np.where(self.uids == ticker[i])[0][0]
                test_logits[date_idx][stock_idx][time_idx] = float(output[i][1])

            end_event.record()
            torch.cuda.synchronize()
            batch_event_time = start_event.elapsed_time(end_event) / 1000
            print(f'Batch Index: {batch_idx + 1}/{len(test_loader)} | Time Use: {batch_event_time} second(s)')

        test_loss_final = test_loss / len(test_loader.sampler)
        test_accuracy_final = (test_acc / len(test_loader.sampler)).cpu().numpy()
        print(f'Test Outcome: Loss - {test_loss_final:.4f} | Accuracy - {test_accuracy_final:.4f}')
        return test_loss_final, test_accuracy_final, test_logits

    def testing_each(self, subperiod_start=None, subperiod_end=None):
        self.model.eval()
        test_loss = 0
        test_acc = 0

        test_loader = torch.utils.data.DataLoader(
            dataset=self.dataset, 
            batch_size=self.config.TRAIN.BATCH_SIZE, 
            shuffle=False
        )

        for batch_idx, (data, ret_short, ret_long, ticker, date_time) in enumerate(test_loader):
            if self.config.TRAIN.LABEL == 'RET_SHORT':
                target = ret_short
            else:
                target = ret_long

            target = (1 - target).unsqueeze(1) @ torch.LongTensor([[1., 0.]]).unsqueeze(1).T + \
                     target.unsqueeze(1) @ torch.LongTensor([[0, 1.]]).unsqueeze(1).T
            target = target.to(torch.float32)

            data, target = data.to(self.device), target.to(self.device)
            output = self.model(data)
            loss = self.criterion(output, target)
            test_loss += loss.item() * data.size(0)
            acc = (output.argmax(1) == target.argmax(1)).sum()
            test_acc += acc

        test_loss_final = test_loss / len(test_loader.dataset)
        test_accuracy_final = (test_acc / len(test_loader.dataset)).cpu().numpy()
        print(f'Sub-period {batch_idx}/{len(test_loader)}: Loss - {test_loss_final}; Accuracy - {test_accuracy_final}')
        return test_loss_final, test_accuracy_final

    def partition(self, start, end, no_partition):
        start_dt = pd.to_datetime(start, format='%Y%m%d')
        end_dt = pd.to_datetime(end, format='%Y%m%d')
        step = (end_dt - start_dt).days // no_partition
        checkpoints = [
            (start_dt + pd.Timedelta(days=x * step)).strftime('%Y%m%d') 
            for x in range(no_partition)
        ] + [end]
        return checkpoints