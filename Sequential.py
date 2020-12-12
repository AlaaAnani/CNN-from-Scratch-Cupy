# %%
import cupy as np
from tqdm.auto import  tqdm, trange
from sklearn.model_selection import train_test_split
import pickle
from sklearn.utils import shuffle

class Sequential():
    def __init__(self):
        self.Layers = []
        self.BATCH_SIZE = 1
        self.stats = None
        self.train_summary = dict()
        self.train_mask = []
        self.lr_mask = []

    def add(self, layer, train=True, lr_ratio=1):
        self.Layers.append(layer)
        self.train_mask.append(train)
        self.lr_mask.append(lr_ratio)

    def insertLayer(self, i, layer, train=True, lr_ratio=1):
        self.Layers.insert(i, layer)
        self.train_mask.insert(i, train)
        self.lr_mask.insert(i, lr_ratio)

    def compile(self):
        out_dim = self.Layers[0].initialize(self.Layers[0].input_shape)
        print(out_dim)
        for i in range(len(self.Layers)):
            if i is 0:
                continue
            out_dim = self.Layers[i].initialize(out_dim)
            print(i, out_dim)

    def pop(self, layer):
        self.Layers = self.Layers[:-1]

    def forward_probagate(self, inputs):
        for i in range(len(self.Layers)):
            activations_i = self.Layers[i].forward(inputs)
            inputs = activations_i
        return activations_i

    def backward_probagate(self, dLdA):
        for i in reversed(range(len(self.Layers))):
            dLdA = self.Layers[i].backward(dLdA)

    def optimize(self, learning_rate, optimizer='SGD', BATCH_SIZE=1, beta1=0.9, beta2=0.99):
        for i in range(len(self.Layers)):
            if self.train_mask[i] is True:
                actual_lr = learning_rate*self.lr_mask[i]
                if optimizer is 'SGD':
                    self.Layers[i].W = self.Layers[i].W - actual_lr*self.Layers[i].dW
                    self.Layers[i].b = self.Layers[i].b - actual_lr*self.Layers[i].db  
                elif optimizer is 'Adam':              
                    self.Layers[i].mo = self.Layers[i].mo*beta1 + (1-beta1)*self.Layers[i].dW
                    self.Layers[i].acc = beta2*self.Layers[i].acc  + (1-beta2)*(self.Layers[i].dW*self.Layers[i].dW)
                    self.Layers[i].W = self.Layers[i].W - actual_lr*self.Layers[i].mo/(np.sqrt(self.Layers[i].acc) + 1e-07)   

                    self.Layers[i].mo_b = self.Layers[i].mo_b*beta1 + (1-beta1)*self.Layers[i].db
                    self.Layers[i].acc_b = beta2*self.Layers[i].acc_b  + (1-beta2)*(self.Layers[i].db*self.Layers[i].db)
                    self.Layers[i].b = self.Layers[i].b - actual_lr*self.Layers[i].mo_b/(np.sqrt(self.Layers[i].acc_b) + 1e-07)           

    @staticmethod
    def batchize(x, BATCH_SIZE=1):
        for i in range(0, len(x), BATCH_SIZE):
            yield x[i:i + BATCH_SIZE]

    def decayed_learning_rate(self, step, decay_rate = 0.96, decay_steps=100000):
        return self.initial_learning_rate * decay_rate ** (step / decay_steps)

    def train(self, loss_func, x_train, y_train,
        oh_comp_func, 
        epochs=50, 
        learning_rate=0.0001, 
        BATCH_SIZE=1,
        validation_split=0.0,
        validation_data=None,
        shuffle_=True,
        initial_epoch=0,
        labels = None,
        optimizer='SGD',
        stats=True,
        x_test=None, y_test=None,
        split_seed=73,
        stats_write=0,
        decay_rate=0.96

        ):
        self.epochs = epochs
        self.loss_func = loss_func
        self.BATCH_SIZE = BATCH_SIZE

        iteration = 0
        correctly_classified = 0
        loss_per_epoch = 0
        batch_count = 0
        self.initial_learning_rate = learning_rate
        if validation_data is None:
            if validation_split != 0:
                if labels is not None:
                    x_train, x_val, y_train, y_val = train_test_split(
                        x_train, y_train, 
                        test_size=validation_split,
                        random_state=split_seed,
                        stratify=labels,
                        shuffle=shuffle_
                        )
                else:
                    x_train, x_val, y_train, y_val = train_test_split(
                        x_train, y_train, 
                        test_size=validation_split,
                        random_state=split_seed,
                        stratify=y_train,
                        shuffle=shuffle_
                        )
            else:
                if shuffle_:
                    print('shuffled no v')
                    x_train, y_train = shuffle(x_train, y_train, random_state=split_seed)           
            
        else:
            x_val, y_val = validation_data[0], validation_data[1]
            if shuffle_:
                x_train, y_train = shuffle(x_train, y_train, random_state=split_seed)           


        x_train = list(self.batchize(x_train, self.BATCH_SIZE))
        y_train = list(self.batchize(y_train, self.BATCH_SIZE))

        x_test = list(self.batchize(x_test, self.BATCH_SIZE))
        y_test = list(self.batchize(y_test, self.BATCH_SIZE))

        x_val = list(self.batchize(x_val, self.BATCH_SIZE))
        y_val = list(self.batchize(y_val, self.BATCH_SIZE))

        if stats is True and initial_epoch == 1:
            self.train_summary['epochs'] = epochs
            self.train_summary['learning_rate'] = learning_rate
            self.train_summary['optimizer'] = optimizer
            self.train_summary['batch_size'] = BATCH_SIZE
            self.train_summary['epochs_i'] = []
            self.train_summary['Train Accuracy'] = []
            self.train_summary['Train Loss'] = []
            self.train_summary['Validation Accuracy'] = []
            self.train_summary['Validation Loss'] = []
            self.train_summary['Test Accuracy'] = []
            self.train_summary['Test Loss'] = []

        self.epoch_i = initial_epoch
        if self.epoch_i == 1:
            self.prev_v_loss = 10e8
            self.prev_t_loss = 10e8
            self.prev_ACCR_ts = 0
            self.step = 0


        for i in range(self.epochs):
            desc = "Epoch "+str(self.epoch_i)
            for x, y in zip(x_train, y_train):
                batch_size = len(x)
                x = np.array(x)
                y = np.array(y)
                y_pred = np.array(self.forward_probagate(x))
                error = loss_func(y, y_pred, deriv=True)
                loss_per_epoch += loss_func(y, y_pred, deriv=False)
                self.backward_probagate(error)
                self.optimize(learning_rate, BATCH_SIZE=batch_size, optimizer=optimizer)
                
                correctly_classified += oh_comp_func(y_pred, y)
                iteration += batch_size
                batch_count += 1
                self.step +=1
            learning_rate = self.decayed_learning_rate(self.step, decay_rate=decay_rate)
            # print(learning_rate)
            loss_per_epoch = loss_per_epoch/batch_count
            epoch_accr = correctly_classified/iteration
            if x_val is not None and y_val is not None:
                _, ACCR, v_loss= self.test(loss_func, x_val, y_val, oh_comp_func)
            
            if x_test is not None and y_test is not None:
                _, ts_ACCR, ts_loss = self.test(loss_func, x_test, y_test, oh_comp_func)
                print("\nEpoch", self.epoch_i, "tr_acc=", epoch_accr, "tr_loss=", loss_per_epoch, "v_acc", ACCR, "v_loss", v_loss, "ts_acc", ts_ACCR, "ts_loss",ts_loss)
            else:
                print("\nEpoch", self.epoch_i, "tr_acc=", epoch_accr, "tr_loss=", loss_per_epoch, "v_acc", ACCR, "v_loss", v_loss)
            


            if stats:
                self.train_summary['epochs_i'].append(self.epoch_i)
                self.train_summary['Train Accuracy'].append(epoch_accr)
                self.train_summary['Train Loss'].append(loss_per_epoch)
                if x_val is not None and y_val is not None:
                    self.train_summary['Validation Accuracy'].append(ACCR)
                    self.train_summary['Validation Loss'].append(v_loss)
                if x_test is not None and y_test is not None:
                    self.train_summary['Test Accuracy'].append(ts_ACCR)
                    self.train_summary['Test Loss'].append(ts_loss)
            # save best model
            if self.epoch_i == 1:
                self.best_model_filename_ts = None
                self.best_model_filename = None
            if stats_write[0] is 1:
                if x_test is not None and y_test is not None:
                    if ts_ACCR > self.prev_ACCR_ts and ts_ACCR > stats_write[1]:
                        self.best_model_filename_ts = "ts_acc"+str(ts_ACCR)+"_ts_loss"+str(ts_loss) 
                        self.save(self.best_model_filename_ts)      
                        self.prev_ACCR_ts = ts_ACCR      
            
                #if x_val is not None and y_val is not None: 
                #    if v_loss < self.prev_v_loss:
                #        self.best_model_filename = "v_acc"+str(ACCR)+"_vloss"+str(v_loss) 
                #        self.save(self.best_model_filename)      
                #        self.prev_v_loss = v_loss
                #else:
                #    if loss_per_epoch < self.prev_t_loss:
                #        self.best_model_filename = "t_acc"+str(epoch_accr)+"_tr_loss"+str(loss_per_epoch) 
                #        self.save(self.best_model_filename)      
                #        self.prev_t_loss = loss_per_epoch               
            # reset epoch params
            iteration = 0
            self.epoch_i +=1
            loss_per_epoch = 0
            correctly_classified = 0
            batch_count = 0
        returns = []
        if self.best_model_filename:
            returns.append(self.load(self.best_model_filename))
        else:
            returns.append(None)

        if self.best_model_filename_ts:
            returns.append(self.load(self.best_model_filename_ts))
        else:
            returns.append(None)  
        
        returns.append(self.train_summary)         
        self.train_summary['epochs'] = self.epoch_i

        return returns
       

    def test(self, loss_func, x_test, y_test, oh_comp_func, tqdm_disable=True):
        self.loss_func = loss_func
        n_samples = 0
        correctly_classified = 0
        avg_loss = 0
        loss_per_batch = 0
        y_pred_list = []
        iteration = 0
        for x, y in zip(x_test, y_test):
            x = np.array(x)
            y = np.array(y)       
            batch_size = x.shape[0]                          
            y_pred = np.array(self.forward_probagate(x))
            y_pred_list.append(y_pred)
            y = np.array(y)
            loss_per_batch += loss_func(y, y_pred, deriv=False)
            correctly_classified += oh_comp_func(y_pred, y)
            n_samples += batch_size
            iteration +=1
        avg_loss = loss_per_batch/iteration
        ACCR = correctly_classified/n_samples
        if tqdm_disable is False:
            print("Correctly classified",correctly_classified, "out of", n_samples, "Test_acc=", ACCR, "Loss=", avg_loss)
        return y_pred_list, ACCR, avg_loss

    def save(self, filename):
        outfile = open('models\\'+ filename,'wb')
        pickle.dump(self, outfile)
        outfile.close()

    def load(self, filename):
        infile = open('models\\'+ str(filename),'rb')
        model_object = pickle.load(infile)
        infile.close()
        return model_object

# %%
