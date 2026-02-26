from __init__ import *
from cnn import *
from torch.utils.data import Dataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.optim.lr_scheduler import CosineAnnealingLR, ExponentialLR, StepLR
import os
import pickle


class StockDataset(Dataset):
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
        with open(pkl_path, 'rb') as p:
            data = pickle.load(p)
        image, ret_long, ret_short = data[timestamp]
        return image, ret_long, ret_short
      
class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience  # Number of epochs, where Loss does not drop as expected
        self.delta = delta        # Minimum expectation of loss drop
        self.counter = 0
        self.best_loss = None
        self.best_acc = None
        self.early_stop_check = False
        self.update_model = False

    def __call__(self, val_loss, val_acc):
        if self.best_loss is None or self.best_acc is None:
            self.best_loss = val_loss
            self.best_acc = val_acc
            self.update_model = True
        elif val_loss > self.best_loss - self.delta or val_acc < self.best_acc + self.delta:  
            # No drop
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop_check = True
                self.update_model = False
        elif val_loss <= self.best_loss - self.delta or val_acc >= self.best_acc + self.delta:  
            # Drop
            self.best_loss = val_loss
            self.best_acc = val_acc
            self.counter = 0
            self.update_model = True
            
class Training(CNN49u, CNN9u):
    def __init__(self, config, GPU_label=0, num_checkpoint=4, delta=0.001):
        super().__init__()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.local_rank = GPU_label
        os.environ["CUDA_VISIBLE_DEVICES"] = str(GPU_label)
        self.no_epoch = config.TRAIN.NEPOCH
        self.valid_ratio = config.TRAIN.VALID_RATIO
        self.batch_size = config.TRAIN.BATCH_SIZE
        self.lr = config.TRAIN.LEARNING_RATE
        self.warmup_epoch = config.TRAIN.WARMUP_EPOCH
        self.base_lr = config.TRAIN.LR_BASE_RATE
        self.weight_decay = config.TRAIN.WEIGHT_DECAY
        self.train_label = config.TRAIN.LABEL
        self.hold_period = config.DATASET.LOOKBACK_WIN
        self.pred_period = config.TRAIN.PREDICT_WIN
        self.edition = config.TRAIN.EDITION
        self.cpu_multi_processing = config.DATASET.PARALLEL_NUM
        self.delta = delta

        self.model_name = config.MODEL
        self.model_save_add = config.TRAIN.MODEL_SAVE_FILE
        self.log_save_add = config.TRAIN.LOG_SAVE_FILE
        self.early_stop = config.TRAIN.EARLY_STOP_EPOCH

        self.project_name = 'Train_{}_{}-Infer_{}_{}'.format(
            config.TRAIN.START_DATE, 
            config.TRAIN.END_DATE, 
            config.INFERENCE.START_DATE, 
            config.INFERENCE.END_DATE
        )
        self.root_dir = os.path.join(config.PATHS.IMAGE_DATA_DIR, self.project_name, 'Train', 'PixelData')

        assert self.model_name in ['CNN49u', 'CNN9u'], (
            f'Invalid Model Name: {self.model_name}. '
            f'Please input either CNN49u for 5-min data or CNN9u for 30-min data.'
        )

        self.dataset = StockDataset(self.root_dir)
        print('Data Retrieval Complete!')
        self.data_size = len(self.dataset)

        self.beforehand(num_checkpoint=num_checkpoint, delta=delta)

        tensorboard_dir = os.path.join(config.PATHS.PROJECT_DIR, 'tensorboard', self.project_name)
        self.tensorboard_runs_dir = os.path.join(tensorboard_dir, self.edition, 'runs')
        os.system('rm -rf {}/*'.format(tensorboard_dir))
        os.makedirs(self.tensorboard_runs_dir, exist_ok=True)

        gc.collect()
        torch.cuda.empty_cache()

    def beforehand(self, num_workers=6, num_checkpoint=4, delta=0.001):
        # This function is used to generate train_loader, valid_loader and their relevant parameters
        self.train_loader_size = int(self.data_size * (1 - self.valid_ratio))
        self.valid_loader_size = self.data_size - self.train_loader_size

        self.train_loader, self.valid_loader = torch.utils.data.random_split(
            self.dataset, 
            [self.train_loader_size, self.valid_loader_size]
        )
        self.train_loader = torch.utils.data.DataLoader(
            dataset=self.train_loader, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=num_workers, 
            pin_memory=True, 
            drop_last=True
        )
        self.valid_loader = torch.utils.data.DataLoader(
            dataset=self.valid_loader, 
            batch_size=self.batch_size, 
            shuffle=False, 
            num_workers=num_workers, 
            pin_memory=True, 
            drop_last=True
        )

        self.valid_checkpoint = [
            i * len(self.train_loader) // num_checkpoint 
            for i in range(1, num_checkpoint + 1)
        ]

        self.early_stopping = EarlyStopping(self.early_stop, delta)

    def cnn_training(self, intra=True, earlystop_mode='intra', lr_scheduler='cosine'):
        self.writer = SummaryWriter(self.tensorboard_runs_dir)
        # Now start training
        if self.model_name == 'CNN49u':
            self.model = CNN49u()
        elif self.model_name == 'CNN9u':
            self.model = CNN9u()

        self.model.to(self.device)

        self.criterion = nn.BCELoss().to(self.device)
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=self.lr, 
            weight_decay=self.weight_decay
        )

        if earlystop_mode == 'fixed':
            if lr_scheduler == 'cosine':
                self.scheduler = CosineAnnealingLR(
                    self.optimizer, 
                    T_max=self.no_epoch - self.warmup_epoch, 
                    eta_min=1e-5
                )
            elif lr_scheduler =='step':
                self.scheduler = StepLR(
                    self.optimizer, 
                    step_size=self.no_epoch // 4
                )
            elif lr_scheduler == 'expo':
                self.scheduler = ExponentialLR(
                    self.optimizer, 
                    gamma=0.95
                )

        print('Ready for Training {}!'.format(self.model_name))

        if intra:
            train_loss_set, valid_loss_set, train_acc_set, valid_acc_set = \
                self.intra_epoch_valid(earlystop_mode=earlystop_mode)
        else:
            train_loss_set, valid_loss_set, train_acc_set, valid_acc_set = self.train_n_epoch()

        log = pd.DataFrame(
            [train_loss_set, train_acc_set, valid_loss_set, valid_acc_set], 
            index=['train_loss', 'train_acc', 'valid_loss', 'valid_acc']
        )
        log.to_csv(self.log_save_add)

        self.writer.close()

    def epoch_iteration(self, mode='train'):
        # sampling_switch = True
        iterate_obj = self.train_loader if mode == 'train' else self.valid_loader
        loss, acc = 0.0, 0.0
        for i, (data, ret_short, ret_long) in enumerate(iterate_obj):
            if self.train_label == 'RET_SHORT':
                target = ret_short
            elif self.train_label == 'RET_LONG':
                target = ret_long
            else:
                pass

            target = (1 - target).unsqueeze(1) @ torch.LongTensor([[1., 0.]]).unsqueeze(1).T + \
                     target.unsqueeze(1) @ torch.LongTensor([[0, 1.]]).unsqueeze(1).T
            target = target.to(torch.float32)

            data, target = data.to(self.device), target.to(self.device)
            # Clear the gradients of all optimized variables in the last iteration. 
            # If this is the first iteration, then initialize these variables.
            self.optimizer.zero_grad()
            # Forward pass: compute predicted output by passing inputs to the model within each iteration
            output = self.model(data)
            # Calculate the batch Loss
            loss = self.criterion(output, target)
            # Backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # A step in Optimization for parameters update
            self.optimizer.step()
            # Record training loss and accuracy
            train_loss += loss.item() * data.size(0)
            acc_batch = (output.argmax(1) == target.argmax(1)).sum()
            train_acc += acc_batch

            self.writer.add_scalars(
                '{} Batch'.format(mode), 
                {'Loss': train_loss, 'No. Accuracy': acc_batch}, 
                (self.epoch_counter - 1) * len(iterate_obj) + i
            )

            # if self.epoch_counter == 1 and sampling_switch:
            #     img_grid = torchvision.utils.make_grid(data[:5])
            #     writer.add_image('Samples', img_grid, self.epoch_counter)

        return train_loss, train_acc

    def multi_valid(self, valid_counter, valid_loss_set, valid_acc_set):
        self.model.eval()
        valid_loss, valid_acc, valid_sample_size = 0.0, 0.0, 0
        with torch.no_grad():
            # Validate on subset of validation loader
            for val_data, val_ret_short, val_ret_long in tqdm(
                self.valid_loader, 
                disable=self.local_rank != 0
            ):
                val_target = val_ret_short if self.train_label == 'RET_SHORT' else val_ret_long
                val_target = (1 - val_target).unsqueeze(1) @ torch.LongTensor([[1., 0.]]).unsqueeze(1).T + \
                             val_target.unsqueeze(1) @ torch.LongTensor([[0, 1.]]).unsqueeze(1).T
                val_target = val_target.to(torch.float32)

                val_data, val_target = val_data.to(self.local_rank), val_target.to(self.local_rank)

                val_output = self.model(val_data)
                val_loss = self.criterion(val_output, val_target)

                valid_loss += val_loss.item() * val_data.size(0)
                valid_acc += (val_output.argmax(1) == val_target.argmax(1)).sum()
                valid_sample_size += val_data.size(0)

            valid_counter += 1

            valid_loss_set.append(valid_loss / valid_sample_size)
            valid_acc_set.append((valid_acc / valid_sample_size).cpu().numpy())
        return valid_counter, valid_loss_set, valid_acc_set

    def intra_epoch_valid(self, earlystop_mode='intra'):
        valid_counter = 0
        valid_loss_record = np.inf
        valid_acc_record = 0
        train_loss_set, valid_loss_set, train_acc_set, valid_acc_set = [], [], [], []

        start_event = torch.cuda.Event(enable_timing=True)
        end_event = torch.cuda.Event(enable_timing=True)

        stop_flag_evt_start = torch.cuda.Event(enable_timing=True)
        stop_flag_evt_end = torch.cuda.Event(enable_timing=True)

        stop_flag = False
        stop_flag_evt_start.record()

        for epoch in range(1, self.no_epoch + 1):
            torch.cuda.synchronize()
            start_event.record()

            self.model.train()

            train_loss, train_acc, train_data_size = 0.0, 0.0, 0.0

            if earlystop_mode == 'intra' and stop_flag:
                stop_flag_evt_end.record()
                whole_time = stop_flag_evt_start.elapsed_time(stop_flag_evt_end) / 1000  
                # Convert the unit from milli-sec to sec
                print(f'Total Complete Epochs: {epoch - 1} | Time Use: {whole_time} second(s)')
                break

            for batch_idx, (data, ret_short, ret_long) in enumerate(self.train_loader):
                # if (batch_idx + 1) % 50 == 0:
                #     print(f'Milestone: Batch Number - {batch_idx + 1}!')
                if earlystop_mode == 'fixed':
                    if epoch <= self.warmup_epoch:
                        warmup_factor = min(
                            1.0, 
                            ((epoch - 1) * len(self.train_loader) + batch_idx + 1) / 
                            (self.warmup_epoch * len(self.train_loader))
                        )
                        for param_group in self.optimizer.param_groups:
                            param_group['lr'] = self.base_lr * warmup_factor

                if self.train_label == 'RET_SHORT':
                    target = ret_short
                elif self.train_label == 'RET_LONG':
                    target = ret_long
                else:
                    pass

                target = (1 - target).unsqueeze(1) @ torch.LongTensor([[1., 0.]]).unsqueeze(1).T + \
                         target.unsqueeze(1) @ torch.LongTensor([[0, 1.]]).unsqueeze(1).T
                target = target.to(torch.float32)

                data, target = data.to(self.device), target.to(self.device)

                self.optimizer.zero_grad()
                output = self.model(data)
                loss = self.criterion(output, target)
                loss.backward()
                self.optimizer.step()

                if earlystop_mode == 'fixed' and epoch > self.warmup_epoch:
                    self.scheduler.step()

                train_loss += loss.item() * data.size(0)
                acc_batch = (output.argmax(1) == target.argmax(1)).sum()
                train_acc += acc_batch
                train_data_size += data.size(0)

                # Validate every N batches
                if (batch_idx + 1) in self.valid_checkpoint:
                    # self.model.eval()
                    # valid_loss, valid_acc, valid_sample_size = 0.0, 0.0, 0
                    # with torch.no_grad():
                    #     # Validate on subset of validation loader
                    #     for i, (val_data, val_ret_short, val_ret_long) in enumerate(self.valid_loader):
                    #         val_target = val_ret_short if self.train_label == 'RET_SHORT' else val_ret_long
                    #         val_target = (1 - val_target).unsqueeze(1) @ torch.LongTensor([[1., 0.]]).unsqueeze(1).T + \
                    #                      val_target.unsqueeze(1) @ torch.LongTensor([[0, 1.]]).unsqueeze(1).T
                    #         val_target = val_target.to(torch.float32)

                    #         val_data, val_target = val_data.to(self.device), val_target.to(self.device)

                    #         val_output = self.model(val_data)
                    #         val_loss = self.criterion(val_output, val_target)

                    #         valid_loss += val_loss.item() * val_data.size(0)
                    #         val_acc_batch = (val_output.argmax(1) == val_target.argmax(1)).sum()
                    #         valid_acc += val_acc_batch
                    #         valid_sample_size += val_data.size(0)

                    #     valid_counter += 1

                    #     valid_loss_set.append(valid_loss / valid_sample_size)
                    #     valid_acc_set.append((valid_acc / valid_sample_size).cpu().numpy())
                    #     print(
                    #         'Epoch: {} - #Validation: {}/{} | Validation Loss: {:.6f} Validation Acc: {:.5f}'
                    #        .format(
                    #             epoch, 
                    #             1 + self.valid_checkpoint.index(batch_idx + 1), 
                    #             len(self.valid_checkpoint), 
                    #             valid_loss_set[-1], 
                    #             valid_acc_set[-1]
                    #         )
                    #     )

                    #     self.writer.add_scalars(
                    #         'Validation All Time', 
                    #         {'Loss': valid_loss_set[-1], 'Accuracy': valid_acc_set[-1]}, 
                    #         valid_counter
                    #     )
                        valid_counter, valid_loss_set, valid_acc_set = \
                            self.multi_valid(valid_counter, valid_loss_set, valid_acc_set)
                        print(
                            'Epoch: {} - #Validation: {}/{} | Validation Loss: {:.6f} Validation Acc: {:.5f}'
                           .format(
                                epoch, 
                                1 + self.valid_checkpoint.index(batch_idx + 1), 
                                len(self.valid_checkpoint), 
                                valid_loss_set[-1], 
                                valid_acc_set[-1]
                            )
                        )

                        self.writer.add_scalars(
                            'Validation All Time', 
                            {'Loss': valid_loss_set[-1], 'Accuracy': valid_acc_set[-1]}, 
                            valid_counter
                        )

                        if earlystop_mode == 'intra':
                            self.early_stopping(valid_loss_set[-1], valid_acc_set[-1])
                            if self.early_stopping.early_stop_check:
                                print(f"Early Stop at Epoch [{epoch}]: Performance hasn't enhanced for {self.early_stop} validations")
                                stop_flag = True
                                break
                            elif self.early_stopping.update_model:
                                print(
                                    'Validation Loss Dropped ({:.6f} --> {:.6f}). | Saving model ...'
                                  .format(valid_loss_record, self.early_stopping.best_loss)
                                )
                                valid_loss_record = self.early_stopping.best_loss
                                torch.save({
                                    'epoch': epoch,
                                    'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict()
                                }, self.model_save_add)
                        elif earlystop_mode == 'fixed':
                            if valid_loss_record - self.delta > valid_loss_set[-1] or valid_acc_record + self.delta < valid_acc_set[-1]:
                                print(
                                    'Validation Loss Dropped ({:.6f} --> {:.6f}). | Saving model ...'
                                  .format(valid_loss_record, valid_loss_set[-1])
                                )
                                valid_loss_record = valid_loss_set[-1]
                                valid_acc_record = valid_acc_set[-1]
                                torch.save({
                                    'epoch': epoch,
                                    'model_state_dict': self.model.state_dict(),
                                    'optimizer_state_dict': self.optimizer.state_dict()
                                }, self.model_save_add)
                        else:
                            pass

            train_loss_set.append(train_loss / train_data_size)
            train_acc_set.append(train_acc / train_data_size)

            self.writer.add_scalar('Complete Epoch Loss - Train', train_loss_set[-1], epoch)
            self.writer.add_scalar('Complete Epoch Accuracy - Train', train_acc_set[-1], epoch)

            print('#Training Epoch {}/{} | Training Loss: {:.6f} Training Acc: {:.5f}'.format(
                        epoch, 
                        self.no_epoch, 
                        train_loss_set[-1], 
                        train_acc_set[-1]
                  )
            )
            
            end_event.record()
            torch.cuda.synchronize()  # Wait for GPU to complete the process
            epoch_time = start_event.elapsed_time(end_event) / 1000  # Convert unit from milli-sec to sec
            print(f'Epoch {epoch} Complete | Time Use: {epoch_time} second(s)')

        return train_loss_set, valid_loss_set, train_acc_set, valid_acc_set

    def train_n_epoch(self):
        valid_loss_min = np.inf
        # Record the training progress
        train_loss_set, valid_loss_set, train_acc_set, valid_acc_set = [], [], [], []

        for epoch in range(1, self.no_epoch + 1):
            # Count the epoch
            self.epoch_counter = epoch
            # Activate model training
            self.model.train()
            train_loss, train_acc = self.epoch_iteration(mode='train')

            self.model.eval()
            valid_loss, valid_acc = self.epoch_iteration(mode='valid')

            len_train = len(self.train_loader.sampler)
            len_valid = len(self.valid_loader.sampler)

            train_loss_set.append(train_loss / len_train)
            valid_loss_set.append(valid_loss / len_valid)

            train_acc_set.append((train_acc / len_train).cpu().numpy())
            valid_acc_set.append((valid_acc / len_valid).cpu().numpy())

            print(
                'Epoch: {} Training Loss: {:.6f} Validation Loss: {:.6f} Training Acc: {:.5f} Validation Acc: {:.5f}'
               .format(epoch, train_loss, valid_loss, train_acc, valid_acc)
            )
            self.writer.add_scalars(
                'Train Epoch', 
                {'Loss': train_loss/len_train, 'Accuracy': train_acc/len_train}, 
                epoch
            )
            self.writer.add_scalars(
                'Valid Epoch', 
                {'Loss': valid_loss/len_valid, 'Accuracy': valid_acc/len_valid}, 
                epoch
            )

            # Save the model, if the validation returns a smaller loss
            if valid_loss < valid_loss_min:
                print(
                    'Validation loss dropped ({:.6f} --> {:.6f}).  Saving model ...'
                   .format(valid_loss_min, valid_loss)
                )
                valid_loss_min = valid_loss
                invalid_epochs = 0  # Counter for epochs when loss did not drop. 
                # Restart counting from 0 when valid loss starts dropping.
                torch.save({
                  'epoch': epoch,
                  'model_state_dict': self.model.state_dict(),
                  'optimizer_state_dict': self.optimizer.state_dict()
                }, self.model_save_add)
            else:
                invalid_epochs += 1

            if invalid_epochs >= self.early_stop:
                print(
                    f"Early Stop at Epoch [{epoch}]: Performance hasn't enhanced for {self.early_stop} epochs"
                )
                break

        return train_loss_set, valid_loss_set, train_acc_set, valid_acc_set

    def loss_acc_plot(self, train_loss_set, train_acc_set, valid_loss_set, valid_acc_set, use_plotly=False):
        loss_acc_dict = {
            'temp train': [train_loss_set, train_acc_set], 
            'temp valid': [valid_loss_set, valid_acc_set]
        }

        if use_plotly:
            pass
        else:
            _, axes = plt.subplots(1, 2, figsize=(20, 6))
            tmp = list(loss_acc_dict.values())
            maxEpoch = len(tmp[0][0])

            maxLoss = max([max(x[0]) for x in loss_acc_dict.values()]) + 0.1
            minLoss = max(0, min([min(x[0]) for x in loss_acc_dict.values()]) - 0.1)

            for name, lossAndAcc in loss_acc_dict.items():
                axes[0].plot(range(1, 1 + maxEpoch), lossAndAcc[0], '-s', label=name)

            axes[0].set_xlabel('Epoch')
            axes[0].set_ylabel('loss')
            axes[0].set_xticks(range(0, maxEpoch + 1, maxEpoch//10))
            axes[0].axis([0, maxEpoch, minLoss, maxLoss])
            axes[0].legend()
            axes[0].set_title("Error")

            maxAcc = min(1, max([max(x[1]) for x in loss_acc_dict.values()]) + 0.1)
            minAcc = max(0, min([min(x[1]) for x in loss_acc_dict.values()]) - 0.1)

            for name, lossAndAcc in loss_acc_dict.items():
                axes[1].plot(range(1, 1 + maxEpoch), lossAndAcc[1], '-s', label=name)

            axes[1].set_xlabel('Epoch')
            axes[1].set_ylabel('Accuracy')
            axes[1].set_xticks(range(0, maxEpoch + 1, maxEpoch//10))
            axes[1].axis([0, maxEpoch, minAcc, maxAcc])
            axes[1].legend()
            axes[1].set_title("Accuracy")

    def count_parameters_with_grad(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)