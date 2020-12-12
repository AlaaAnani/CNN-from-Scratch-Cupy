# %%
import cupy as np
from Layers.ConvLayer2D import Conv2D
from Layers.Dense import Dense
from Layers.Flatten import Flatten
from Layers.MaxPool import MaxPool
from Sequential import Sequential
import os
from utils.nn_utils import SoftMax, oh_p_CCRN, graph_scatter, graph_bar, LeakyRelU, CrossEntropyLoss, oh_p_comp, get_dataset, batchize
from sklearn.preprocessing import OneHotEncoder

os.environ["CUDA_VISIBLE_DEVICES"] = "0,2"
dataset_folder = "flower_photos/flower_photos"
labels = ['daisy', 'dandelion', 'roses', 'sunflowers', 'tulips']
np_data_files = ["x_train", "y_train", "x_test", "y_test"]
scores_folder_name = "scores_RGB"
np_dataset_folder_name = "np_dataset/RGB_64_cleaned_augmented"
folders = [np_dataset_folder_name, scores_folder_name]


################### FOR USER ###################################
# # if you wish to merely load the pickle file of the best model
# # set TRAIN=False and run the code. 
# # If you wish to train, set TRAIN = True and specify the parameters
# # of the model.Train function as you wish. (It is submitted as the best model config)
IMG_WIDTH = 64
IMG_HEIGHT = 64
TEST_SET_SIZE = 100
BEST_MODEL = "TS_ACC77"
TRAIN = True
BATCH_SIZE = 96
RANDOM_SEED = 3737
FIRST_RUN = 0
AUGMENT = False
np.random.seed(RANDOM_SEED)
################################################################

enc = OneHotEncoder()
oh_classes = enc.fit_transform(np.reshape(np.array(range(5)).get(), (5,1))).toarray()
oh_classes_dict = {}
for i in range(5):
    oh_classes_dict[labels[i]] = oh_classes[i]


x_train, y_train, x_test, y_test = get_dataset(
    FIRST_RUN, 
    dataset_folder, 
    labels, 
    oh_classes_dict,
    IMG_WIDTH, IMG_HEIGHT,
    TEST_SET_SIZE, 0, 
    np_dataset_folder_name, np_data_files,
    augment=AUGMENT
    )

y_train_int_labels = enc.inverse_transform(y_train.get())
y_test_int_labels = enc.inverse_transform(y_test.get())
x_train = x_train.reshape(x_train.shape[0], IMG_WIDTH, IMG_WIDTH, 3)
x_test = x_test.reshape(x_test.shape[0], IMG_WIDTH, IMG_WIDTH, 3)


# %%
model = Sequential()
model.add(Conv2D((3, 3), 64, LeakyRelU, padding='same', input_shape=(BATCH_SIZE, IMG_HEIGHT, IMG_WIDTH, 3), first=True))
model.add(MaxPool((5, 5), 5))
model.add(Conv2D((3, 3), 64, LeakyRelU, padding='same'))
model.add(MaxPool((5, 5), 5))
model.add(Conv2D((5, 5), 96, LeakyRelU, padding='same'))
model.add(Flatten())
model.add(Dense(512, LeakyRelU), lr_ratio=0.001)
model.add(Dense(512, LeakyRelU), lr_ratio=0.001)
model.add(Dense(5, SoftMax))

model.compile()
# %%
if TRAIN:
    model_data = model.train(
        loss_func=CrossEntropyLoss, 
        x_train=x_train, 
        y_train=y_train, 
        oh_comp_func=oh_p_comp, 
        epochs=150, 
        learning_rate=0.0009,
        shuffle_=True,
        validation_split=0.05,
        initial_epoch=1,
        BATCH_SIZE=BATCH_SIZE,
        labels=y_train_int_labels,
        optimizer='Adam',
        stats=True,
        x_test=x_test, y_test=y_test,
        stats_write=(1, 0.77),
        decay_rate = 0.96
        )
    best_model_v = model_data[0] # best model according to validation loss
    best_model_t = model_data[1] # best model according to test ACCR
    train_summary = model_data[2] # summary
else:
    model = model.load(BEST_MODEL)
    # make the test data batched 
    x_test = batchize(x_test, BATCH_SIZE)
    y_test = batchize(y_test, BATCH_SIZE)

    y_pred_list, ACCR, avg_loss = model.test(
        loss_func=CrossEntropyLoss, 
        x_test=x_test, 
        y_test=y_test, 
        oh_comp_func=oh_p_comp
        )
    print("Model ACCR on Test=", ACCR)
    print("Model Loss on Test=", avg_loss)

    # debatchize the prediction list
    pred_list = []
    for i, y_b in enumerate(y_pred_list):
        for _, sample in enumerate(y_pred_list[i]):
            pred_list.append([sample])
    y_pred_list = pred_list
# %%
# LOSS Graphs
# # extract axes
MODEL_NAME = '77' # To use for naming the graphs of the model after the training

x_axis = np.array(model.train_summary['epochs_i'])
y_axis1 = np.array(model.train_summary['Validation Loss'])
y_axis2 = np.array(model.train_summary['Train Loss'])
y_axis3 = np.array(model.train_summary['Test Loss'])

# # get annotations 
best_epoch = np.array(model.train_summary['Test Loss']).argmin()
best_loss = np.min(np.array(model.train_summary['Test Loss']))
pointer = "Best Test Loss= " + str(best_loss)[0:7]
annotation = [(best_epoch, best_loss, pointer)]

best_epoch = np.array(model.train_summary['Validation Loss']).argmin()
best_loss = np.min(np.array(model.train_summary['Validation Loss']))
pointer = "Best Validation Loss= " + str(best_loss)[0:7]
annotation.append((best_epoch, best_loss, pointer))

graph_scatter(
    x=[x_axis, x_axis, x_axis], 
    y=[y_axis1, y_axis2, y_axis3], 
    x_title='Epochs', y_title="Cross Entropy Average Loss",\
    labels=["Validation Loss", "Train Loss", "Test Loss"], 
    graphs_folder_name="Graphs", 
    graph_title="Training vs. Validation Vs. Test Loss", 
    graph_file_name="tr_v_ts_"+MODEL_NAME, 
    write=1,
    annotation=annotation
    )

graph_scatter(
    x=[x_axis, x_axis], 
    y=[y_axis1, y_axis2], 
    x_title='Epochs', y_title="Cross Entropy Average Loss",\
    labels=["Validation Loss", "Train Loss"], 
    graphs_folder_name="Graphs", 
    graph_title="Training vs. Validation Loss", 
    graph_file_name="tr_v_"+MODEL_NAME, 
    write=1,
    annotation=[annotation[1]]
    )

# ACCR Graph
x_axis = np.array(model.train_summary['epochs_i'])
y_axis1 = np.array(model.train_summary['Validation Accuracy'])
y_axis2 = np.array(model.train_summary['Train Accuracy'])
y_axis3 = np.array(model.train_summary['Test Accuracy'])

annotation = []
best_epoch = np.array(model.train_summary['Test Accuracy']).argmax()
best_ts_accr = np.max(np.array(model.train_summary['Test Accuracy']))
pointer = "Best Test ACCR = " + str(best_ts_accr)[0:5]
annotation=[(best_epoch, best_ts_accr, pointer)]

graph_scatter(
    x=[x_axis, x_axis, x_axis], 
    y=[y_axis1, y_axis2, y_axis3], 
    x_title='Epochs', y_title="Cross Entropy Average Loss",\
    labels=["Validation Accuracy", "Train Accuracy", "Test Accuracy"], 
    graphs_folder_name="Graphs", 
    graph_title="Training vs. Validation Vs. Test ACCR (Best Epoch="+str(best_epoch)+")", 
    graph_file_name="tr_v_ts_ACCR"+MODEL_NAME, 
    write=1,
    annotation=annotation
    )

# Graph CCRN
CCRN = oh_p_CCRN(y_pred_list, y_test_int_labels, labels)
ccrn_vals = list(CCRN.values())
ccrn_val = []
for i in range(len(ccrn_vals)):
    x = ccrn_vals[i].get()
    ccrn_val.append(x)

graph_bar(x=list(CCRN.keys()), y=ccrn_val,
    graph_title = "Correct Classification Rate (CCRn) of All Classes",
    graph_file_name = "ccrn_"+MODEL_NAME,
    x_title = "Classes",
    y_title = "CCRn",
    graphs_folder_name="Graphs"
    )
