from __init__ import *
import initialsetting as IS

import utils
reload(utils)

class Data():
    def __init__(self, yaml_dir, commoncache_dir):
        self.config_dir = yaml_dir
        self.commoncache_dir = commoncache_dir

        self.preset = IS.InitialSetting(self.config_dir)

        self.train_start = self.preset.config.TRAIN.START_DATE
        self.train_end = self.preset.config.TRAIN.END_DATE
        self.infer_start = self.preset.config.INFERENCE.START_DATE
        self.infer_end = self.preset.config.INFERENCE.END_DATE

        self.Open_dir = os.path.join(self.commoncache_dir, 'IntervalFull/IntervalFull.open.N49,5184f')
        self.High_dir = os.path.join(self.commoncache_dir, 'IntervalFull/IntervalFull.high.N49,5184f')
        self.Low_dir = os.path.join(self.commoncache_dir, 'IntervalFull/IntervalFull.low.N49,5184f')
        self.Close_dir = os.path.join(self.commoncache_dir, 'IntervalFull/IntervalFull.close.N49,5184f')
        self.Volume_dir = os.path.join(self.commoncache_dir, 'IntervalFull/IntervalFull.volume.N49,5184f')

        self.AShareNST_dir = os.path.join(self.commoncache_dir, 'AShareNST/AShareNST.N,5184c')
        self.Adjust_dir = os.path.join(self.commoncache_dir, 'ForwardAdjPrices/ForwardAdjPrices.S_DQ_ADJFACTOR.N,5184f')
        self.UpLim_dir = os.path.join(self.commoncache_dir, 'Limits/UplimPrice.N,5184f')
        self.DnLim_dir = os.path.join(self.commoncache_dir, 'Limits/DnlimPrice.N,5184f')

        self.tradedays_dir = os.path.join(self.commoncache_dir, '_universe/dates.NI')
        self.uid_dir = os.path.join(self.commoncache_dir, '_universe/uid.N128C')

        self.tradeday = np.memmap(self.tradedays_dir, mode = 'r', dtype = self.dtype_map(self.tradedays_dir[-1]))
        self.allday_len = len(self.tradeday)
        self.uids = np.memmap(self.uid_dir, mode = 'r', dtype = self.dtype_map(self.uid_dir[-4:]))
        self.uid_len = len(self.uids)

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

    def retrieve(self, startdate, enddate, freq='5min'):
        day_mask = (self.tradeday >= startdate) & (self.tradeday <= enddate)
        self.trade_len = sum(day_mask)

        if freq == '5min':
            df_open, df_high, df_low, df_close, df_volume = \
                np.memmap(self.Open_dir, mode='r', dtype=self.dtype_map(self.Open_dir[-1]), 
                          shape=(self.allday_len, int(self.Open_dir[-5: -1]), int(self.Open_dir[-8: -6])))[day_mask, :self.uid_len], \
                np.memmap(self.High_dir, mode='r', dtype=self.dtype_map(self.High_dir[-1]), 
                          shape=(self.allday_len, int(self.High_dir[-5: -1]), int(self.High_dir[-8: -6])))[day_mask, :self.uid_len], \
                np.memmap(self.Low_dir, mode='r', dtype=self.dtype_map(self.Low_dir[-1]), 
                          shape=(self.allday_len, int(self.Low_dir[-5: -1]), int(self.Low_dir[-8: -6])))[day_mask, :self.uid_len], \
                np.memmap(self.Close_dir, mode='r', dtype=self.dtype_map(self.Close_dir[-1]), 
                          shape=(self.allday_len, int(self.Close_dir[-5: -1]), int(self.Close_dir[-8: -6])))[day_mask, :self.uid_len], \
                np.memmap(self.Volume_dir, mode='r', dtype=self.dtype_map(self.Volume_dir[-1]), 
                          shape=(self.allday_len, int(self.Volume_dir[-5: -1]), int(self.Volume_dir[-8: -6])))[day_mask, :self.uid_len]

            self.no_interval = int(self.Close_dir[-8: -6])
            if self.no_interval == 49:
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

            df_ashare, df_uplim, df_dnlim = \
                np.memmap(self.AShareNST_dir, mode='r', dtype=self.dtype_map(self.AShareNST_dir[-1]), 
                          shape=(self.allday_len, int(self.AShareNST_dir[-5: -1])))[day_mask, :self.uid_len], \
                np.memmap(self.UpLim_dir, mode='r', dtype=self.dtype_map(self.UpLim_dir[-1]), 
                          shape=(self.allday_len, int(self.UpLim_dir[-5: -1])))[day_mask, :self.uid_len], \
                np.memmap(self.DnLim_dir, mode='r', dtype=self.dtype_map(self.DnLim_dir[-1]), 
                          shape=(self.allday_len, int(self.DnLim_dir[-5: -1])))[day_mask, :self.uid_len]

            df_adj = np.memmap(self.Adjust_dir, mode='r', dtype=self.dtype_map(self.Adjust_dir[-1]), 
                              shape=(self.allday_len, int(self.Adjust_dir[-5: -1])))[day_mask, :self.uid_len]

            new_df_uplim, new_df_dnlim, new_df_ashare = \
                np.repeat(df_uplim[:, :, np.newaxis], self.no_interval, axis=2), \
                np.repeat(df_dnlim[:, :, np.newaxis], self.no_interval, axis=2), \
                np.repeat(df_ashare[:, :, np.newaxis], self.no_interval, axis=2)

            UpLimIndicator, DnLimIndicator = \
                ~(abs(new_df_uplim - df_close) < 0.0001), \
                ~(abs(new_df_dnlim - df_close) < 0.0001)

            df_adjopen, df_adjhigh, df_adjlow, df_adjclose = df_open * np.repeat(df_adj[:, :, np.newaxis], df_open.shape[2], axis=2), \
                                                            df_high * np.repeat(df_adj[:, :, np.newaxis], df_high.shape[2], axis=2), \
                                                            df_low * np.repeat(df_adj[:, :, np.newaxis], df_low.shape[2], axis=2), \
                                                            df_close * np.repeat(df_adj[:, :, np.newaxis], df_close.shape[2], axis=2)

            all_df_ashare = new_df_ashare.reshape(-1)

            self_df_adj0, self_df_adjH, self_df_adjL, self_df_adjC = df_adjopen.reshape(-1)[all_df_ashare], \
                                                                    df_adjhigh.reshape(-1)[all_df_ashare], \
                                                                    df_adjlow.reshape(-1)[all_df_ashare], \
                                                                    df_adjclose.reshape(-1)[all_df_ashare]

            self_df_volume, self_df_notuplim, self_df_notdnlim = df_volume.reshape(-1)[all_df_ashare], \
                                                                UpLimIndicator.reshape(-1)[all_df_ashare], \
                                                                DnLimIndicator.reshape(-1)[all_df_ashare]

            self_date_idx = np.repeat(self.tradeday[day_mask].astype(str), self.uid_len * self.no_interval)[all_df_ashare]
            self.ticker_idx = np.tile(np.repeat(self.uids, self.no_interval), self.trade_len)[all_df_ashare]
            self.time_idx = np.tile(np.array(self.intervals), self.uid_len * self.trade_len)[all_df_ashare]
            self.ashare_list = np.unique(self.ticker_idx)

class Imaging(Data):
    def __init__(self, setting, yaml_dir, short=1, freq='5min'):
        # setting should be a namedtuple generated by the class 'InitialSetting()'
        Data.__init__(self, yaml_dir=yaml_dir, commoncache_dir=setting.PATHS.COMMONCACHE_DIR)
        self.price_lab = ['Open', 'High', 'Low', 'Close']
        self.volume_lab = ['Volume']
        self.left = ['Vwap']
        self.support_indicator = []
        self.save_img_dir = setting.PATHS.IMAGE_DATA_DIR
        
        # self.raw_data_directory = '/dfs/data/Project/data/SparkData/df/Train_().().Test_().().Infer_().().format(setting.TRAIN_START_DATE, setting.TRAIN_END_DATE, setting.TEST_START_DATE, setting.TEST_END_DATE, setting_INFERENCE_START_DATE, setting_INFERENCE_END_DATE)
        
        # For use of day number calculation
        if freq == '5min':
            self.day_denominator = 49
        elif freq == '30min': 
            self.day_denominator = 9

        self.project_name = 'Train_().().Infer_().().'.format(setting.TRAIN.START_DATE, setting.TRAIN.END_DATE, setting.INFERENCE.START_DATE, setting.INFERENCE.END_DATE)

        # Now create self attributes within DATASET by setattr()
        self.config_dict = setting.DATASET._asdict()
        for config, value in self.config_dict.items():
            # attribute LOOKBACK_WIN -> int: the window for data to be included later in pixel image
            # attribute START_DATE -> int: start time of the operation (train, test, inference)
            # attribute END_DATE -> int: end time of the operation (train, test, inference)
            # attribute INDICATIONS -> list of namedtuple objects: with two elements 'NAME', 'PARAM'.
            # attribute SHOW_VOLUME -> boolean: True for the volume data to be shown, together with price data; otherwise, only volume data.
            # attribute SAMPLE_RATE -> float: a probability under which some images will not be generated to achieve a randomness purpose
            # attribute PARALLEL_NUN -> int: the number of CPU cores that will be used; note that -1 for all.
            setattr(self, config, value)

        # ! Warning Checks for Parameter Inputs
        assert self.LOOKBACK_WIN in [9, 49], f'Window Size is required to be larger or equal to 49 or 9.'

        self.short_win = short
        self.long_win = setting.TRAIN.PREDICT_WIN
        self.short_label = 'ret()'.format(self.short_win)
        self.long_label = 'ret()'.format(self.long_win)
        
        # For CNN Label use
        self.label = setting.TRAIN.LABEL
        if self.label == 'RET_SHORT':
            self.pred_win = self.short_win
        else:
            self.pred_win = self.long_win
          
        if self.LOOKBACK_WIN == 49:
            # ! image_size -> tuple: the size of image generated by this function, with Length at Least (49 * 3) for 5-min bar data
            self.image_size = (49 * 3, 49 * 3)
        elif self.LOOKBACK_WIN == 9:
            self.image_size = (27, 27)

        self.padding_days = self.LOOKBACK_WIN // self.day_denominator + 1

        self.bug_tickers = []

    def bloody_bug(self, ticker, mode, volume=False, use_plotly=False):
        self.bind_cpu()
        ticker_mask = (self.ticker_idx == ticker)
        df_o, df_h, df_l, df_c, df_v, df_notuplim, df_notdnlim, didx, tidx = self.df_adjO[ticker_mask], \
                                                                              self.df_adjH[ticker_mask], \
                                                                              self.df_adjL[ticker_mask], \
                                                                              self.df_adjC[ticker_mask], \
                                                                              self.df_volume[ticker_mask], \
                                                                              self.df_notuplim[ticker_mask], \
                                                                              self.df_notdnlim[ticker_mask], \
                                                                              self.date_idx[ticker_mask], \
                                                                              self.time_idx[ticker_mask]

        try:
            self.data2image(df_o, df_h, df_l, df_c, df_v, df_notuplim, df_notdnlim, didx, tidx, ticker, mode, volume, use_plotly)
        except:
            print(ticker)
            self.bug_tickers.append(ticker)

    def data2image(self, df_o, df_h, df_l, df_c, df_v, df_notuplim, df_notdnlim, didx, tidx, ticker, mode='train', volume=False, use_plotly=False, save_size=200):
        # | df --> memmap: an aggregated numpy arrays with multiple columns including ONLCV + ret1 + ret50 of ONE TICKER, i.e.
        # memmap_file_name[memmap_file_name['Ticker'] == ticker],
        # where tqdm iterate over ticker
        if len(df_c) < self.day_denominator + self.long_win + 1:
            return

        # Cut the tail of the df by 50, since they are all np.nan
        df_ret_long, df_ret_short = (df_c[self.long_win + 1:] - df_c[:-self.long_win - 1]) / df_c[:-self.long_win - 1], \
                                    (df_c[self.short_win + 1:] - df_c[:-self.short_win - 1]) / df_c[:-self.short_win - 1]

        df_o, df_h, df_l, df_c, df_v, df_ret_short, df_uplim, df_dnlim = df_o[:len(df_ret_long)], \
                                                                        df_h[:len(df_ret_long)], \
                                                                        df_l[:len(df_ret_long)], \
                                                                        df_c[:len(df_ret_long)], \
                                                                        df_v[:len(df_ret_long)], \
                                                                        df_ret_short[:len(df_ret_long)], \
                                                                        df_notuplim[:len(df_ret_long)], \
                                                                        df_notdnlim[:len(df_ret_long)]

        l = len(df_c)

        # 1. Initialize required lists, indicator, dataset, valid_time, etc
        if self.DATASET.INDICATORS is not None:
            for ti_name in self.DATASET.INDICATORS._fields:
                if ti_name == 'MA':
                    ma = getattr(self.DATASET.INDICATORS, ti_name)
                    for param in ma._fields:
                        ma_win = getattr(ma, param)
          
        indicator = (df_c[ma_win:] - df_c[:len(df_c) - ma_win]) / df_c[:len(df_c) - ma_win]

        # dataset = []

        missing_mask = np.where(np.isnan(df_c))[0]

        if len(missing_mask) == 0:
            miss_idx = 0
            marker = -1
            mask_end = -1
        else:
            miss_idx = 0
            marker = missing_mask[miss_idx]
            mask_end = missing_mask[-1]

        # 2.1 Iterate on data to generate subset of the data for imaging plotting use
        #     In this block, the output should be image dataset (pixel image array, tech indicator labels, say ret5, ret20) for training and testing;
        #     ticker, its dataset above, valid time for inference/evaluation.

        # Initial Setting
        d = self.LOOKBACK_WIN - 1
        counter = 0
        entry = {}
        class_stat = {'Ticker': ticker, 'Long': 0, 'Short': 0, 'Both': 0, 'Neither': 0}

        while d < l: 
            time_label = (didx[d].item())[:4] + '-' + (didx[d].item())[4:6] + '-' + (didx[d].item())[6:] + '-' + (tidx[d].item())
            
            # Make sure the last timestamp of the time cut later than the start datetime
            # Randomly skip some trading days to avoid overfitting
            if np.random.rand(1) > self.SAMPLE_RATE and mode == 'train':
                # I why do we need this in 'inference' mode?
                d += 1
                continue
            
            # ? Label the image data by the last (date + time) of the pixel image
            if d >= marker and marker != mask_end:
                while missing_mask[miss_idx + 1] - marker <= self.LOOKBACK_WIN:
                    miss_idx += 1
                    marker = missing_mask[miss_idx]
                    if marker == mask_end:
                        break
                d = marker + self.LOOKBACK_WIN
                
                if marker != mask_end:
                    marker = missing_mask[miss_idx + 1]
                    miss_idx += 1
                    
            elif marker <= d < self.LOOKBACK_WIN and marker == mask_end:
                d += self.LOOKBACK_WIN
            elif d == mask_end:
                marker = mask_end
                d += self.LOOKBACK_WIN
            else:
                # missing_slice = data_cut[d-(self.LOOKBACK_WIN - 1): d+1][self.missing_lab].reset_index(drop=True)
                price_slice = (df_o[d-(self.LOOKBACK_WIN - 1):d+1], 
                              df_h[d-(self.LOOKBACK_WIN - 1):d+1],
                              df_l[d-(self.LOOKBACK_WIN - 1):d+1],
                              df_c[d-(self.LOOKBACK_WIN - 1):d+1])
                
                if True in (na_indicator := np.isnan(price_slice[-1])):
                    d += np.where(na_indicator)[0][-1] + 1
                    continue
                    
                volume_slice = df_v[d-(self.LOOKBACK_WIN - 1):d+1]  # ! Need an if-statement to decide if volume is shown in the image
                lim_slice = (df_uplim[d-(self.LOOKBACK_WIN - 1):d+1],
                            df_dnlim[d-(self.LOOKBACK_WIN - 1):d+1])

                # Ensure there is no missing data within the Moving_Window
                
                if use_plotly:
                  pass
                else:
                  image = self.image_data(price_slice, volume_slice, lim_slice, indicator = [], volume = False)
                  
                # ? Label Setting for CNN Classification Use
                # ? Could use other tech indicators as labels
                label_short = 1 if np.sign(df_ret_short[d]) > 0 else 0
                label_long  = 1 if np.sign(df_ret_long[d])  < 0 else 0
                class_stat['Long'] += label_long
                class_stat['Short'] += label_short
                
                if label_short == 1 and label_long == 1:
                  class_stat['Both'] += 1
                if label_short == 0 and label_long == 0:
                  class_stat['Neither'] += 1
                  
                entry[time_label] = [image, label_short, label_long]
                counter += 1
                # Under each iteration, the data to be saved is a list of image as the first element, then binary data of ret_short and ret_long
                
                d += 1
                if counter == save_size:
                    self.save_single_data(entry, ticker, mode, save_mode = 'data')
                    counter = 0
                    entry = {}
        
        if 0 < len(entry.keys()) < save_size:
            self.save_single_data(entry, ticker, mode, save_mode = 'data')
        
        return class_stat
    
    def save_single_data(self, entry, ticker, mode, save_mode='data'):
        if save_mode == 'data':
            new_dir = self.save_img_dir + '{}/{}/Pixel{}/{}/'.format(
                self.project_name, 
                mode.capitalize(), 
                save_mode.capitalize(), 
                ticker
            )
            os.makedirs(new_dir, exist_ok=True)
            time_labels = list(entry.keys())
            with open(new_dir + '{}_to_{}.pkl'.format(time_labels[0], time_labels[-1]), 'wb') as p:
                pickle.dump(entry, p)
        elif save_mode == 'image':
            pass
        else:
            pass
          
    def image_data(self, price_slice, volume_slice, lim_slice, indicator=[], volume=False):
        # Rescaling in PV data for pixel plotting.
        # ! The code here is the reason of pd.errors.IntCastingNaNError
        image = np.zeros(self.image_size, dtype=np.uint8)

        min_values = np.array([np.min(col) for _, col in enumerate(price_slice)])
        max_values = np.array([np.max(col) for _, col in enumerate(price_slice)])

        updn_label = np.array([1 if False in col else 0 for _, col in enumerate(lim_slice)])

        updn_lim = np.equal(min_values, max_values)
        # Convert the tuple of arrays to a numpy array
        price_slice = np.array(np.array(price_slice.tolist())).T
        volume_slice = np.array(volume_slice.tolist())
        volume_slice = (volume_slice - np.min(volume_slice)) / (np.max(volume_slice) - np.min(volume_slice))

        if sum(updn_lim) == 4:
            if updn_label[0]:  # Indicate Uplim
                filling_lvl = 3/4
            elif updn_label[1]:  # Indicate Dnlim
                filling_lvl = 1/4
            price_slice = np.full(price_slice.shape, filling_lvl)
        elif 0 < sum(updn_lim) < 4:
            rescale = max_values[-1] - min_values[-1]
            rescale_idx = np.where(np.equal(min_values, max_values))
            for idx, arr in enumerate(price_slice):
                if idx in rescale_idx:
                    price_slice[idx] = (arr - min_values[-1]) / rescale
                else:
                    price_slice[idx] = (arr - min_values[idx]) / (max_values[idx] - min_values[idx])
        else:
            price_slice = (price_slice - min_values) / (max_values - min_values)

        # Recall the size of pixel image is (49 * 3, 49 * 3)
        if not volume:
            price_slice = (price_slice * (self.image_size[0] - 1)).astype(int) 
            # lambda x: x * self.image_size[0] - 1
        else:
            pass
          
        for i in range(len(price_slice)):
            # Manual OHLC Plot
            # ? Note: 255 here means color white; otherwise, 0 for black
            # Visually, the proper image is the transpose of the below image to be generated
            image[price_slice[i][0], i * 3] = 255
            image[price_slice[i][2]: price_slice[i][1] + 1, i * 3 + 1] = 255
            image[price_slice[i][3], i * 3 + 2] = 255

            for ind in range(len(indicator)):
                # iterate over the names of tech indicators
                image[price_slice[i][4 + ind], i * 3 : i * 3 + 2] = 255

            if volume:
                image[: volume_slice[i][0], i * 3 + 1] = 255

        return image

    def plotly_data(self, price_slice, volume_slice, indicator = [], volume = False, mode = 'train'):
        pass
      
    @utils.timer('Image Dataset', '12')
    def all_data(self, modes=['train', 'infer', 'train_all'], freq='5min'):
        # Fix the seed
        np.random.seed(69)
        for mode in modes:
            if mode == 'train' or mode == 'train_all':
                self.retrieve(startdate=self.train_start, enddate=self.train_end, freq=freq)
            if mode == 'infer':
                self.retrieve(startdate=self.infer_start, enddate=self.infer_end, freq=freq)

            if os.path.exists(self.save_img_dir + self.project_name + '/{}/PixelData'.format(mode)):
                print(f'{mode.capitalize()} Data were Already Generated!')
            else:
                # Alternatively: class_stat = self.multi_threads(mode)
                class_stat = Parallel(n_jobs=self.PARALLEL_NUM, backend='loky')(
                    delayed(self.bloody_bug)(
                        ticker=g, mode=mode, volume=False, use_plotly=False
                    )
                    for g in tqdm.tqdm(
                        self.ashare_list,
                        desc=f'Generating Pixel Dataset for Training (sample rate: {self.SAMPLE_RATE})' 
                        if mode == 'train' 
                        else f'Generating Pixel Dataset for Inference' 
                        if mode == 'infer' 
                        else f'Generating All Pixel Dataset for Training' 
                        if mode == 'train_all' 
                        else 'TBC, otherwise...'
                    )
                )

                with open(self.save_img_dir + self.project_name + '/{}_bug_tickers.pkl'.format(mode), 'wb') as p:
                    pickle.dump(self.bug_tickers, p)

                with open(self.save_img_dir + self.project_name + '/{}_class_stat.pkl'.format(mode), 'wb') as p:
                    pickle.dump(class_stat, p)

                print(f'Mission Complete: Generating {mode.capitalize()} Data!')
                
    def bind_cpu(self):
        p = psutil.Process()
        available_cores = list(range(psutil.cpu_count()))
        p.cpu_affinity([available_cores[len(p.cpu_affinity()) % len(available_cores)]])


    def multi_threads(self, mode):
        max_workers = self.PARALLEL_NUM or psutil.cpu_count()
        process_func = partial(self.bloody_bug, mode=mode, volume=False, use_plotly=False)

        with Pool(max_workers) as pool:
            results = list(
                tqdm(
                    pool.imap(process_func, self.ashare_list),
                    total=len(self.ashare_list),
                    desc=f'Generating Pixel Dataset for Training (sample rate: {self.SAMPLE_RATE})' 
                    if mode == 'train' 
                    else 'Generating Pixel Dataset for Inference'
                )
            )

        return results


