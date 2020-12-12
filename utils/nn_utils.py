# %%
import os
import pathlib
import statistics
from statistics import mode
from tqdm import tqdm 
import matplotlib.pyplot as plt
import pandas as pd
from skimage.io import imread
from skimage.transform import resize, rotate
from skimage.filters import median
from skimage import exposure
from numpy import fliplr, flipud
import cupy as np
from skimage import exposure
from tqdm import tqdm_notebook
import pickle
import plotly.graph_objects as go
def batchize(x, BATCH_SIZE=1):
    for i in range(0, len(x), BATCH_SIZE):
        yield x[i:i + BATCH_SIZE]
def oh_p_comp(y_pred, y):
    return np.sum(1*(np.argmax(y_pred, axis=1) == np.argmax(y, axis=1)))
def oh_p_CCRN(y_pred_list, y_test_labelled, labels):
    accuracy_per_class = dict()
    y_t = [y[0] for y in y_test_labelled]
    y_t = np.array(y_t)
    y_p = [y[0] for y in y_pred_list]
    y_p = np.argmax( np.array(y_p), axis=1)

    for label in labels:
        label_code = labels.index(label)
        label_codes = np.full(y_test_labelled.shape[0], label_code)
        accuracy_per_class[label] = np.sum((y_p == y_t) & (label_codes == y_p))/100.0
    return accuracy_per_class

def SoftMax(x, deriv=False):
    x = np.array(x)
    if deriv is False:
        max_act = np.max(x, axis=1)
        x = x - max_act[:, np.newaxis]
        x = np.exp(x)
        sig_ex = np.sum(x, axis=1).T
        sig_ex = np.array([sig_ex] * x[0].shape[0])
        sig_ex = np.transpose(sig_ex)
        res = x/sig_ex
        #y = np.exp(x - np.max(x, axis=1, keepdims=True))
        #y = y / np.sum(y, axis=1, keepdims=True)
        return res
    return x*(1-x)
    # return jac
def LeakyRelU(x, alpha=0.07, deriv=False):
    x = np.array(x)
    if deriv is False:
        return np.maximum(alpha*x, x)
    x[x<=0] = alpha
    x[x>0] = 1
    return x

def RelU(x, deriv=False):
    x = np.array(x)
    if deriv is False:
        return np.maximum(0, x)
    x[x<=0] = 0
    x[x>0] = 1
    return x

def CrossEntropyLoss(oh_target, y_pred, deriv=False):
    if deriv is False:
        oh_target = np.array(oh_target)
        y = -np.log(y_pred.T)
        loss = np.mean(oh_target.dot(y).diagonal())
        #batch_size = y_pred.shape[0]
        #loss = -1 / batch_size * (oh_target * np.log(np.clip(y_pred, 1e-7, 1.0))).sum()
        return loss
    return y_pred - oh_target
    #return  - np.divide(oh_target, np.clip(y_pred, 1e-7, 1.0))
def get_image_augmentations(img, IMG_WIDTH, IMG_HEIGHT):
    img_augmentations_arr = []
    img_augmentations_arr.append(np.array(resize(rotate(img, angle=90), (IMG_WIDTH, IMG_HEIGHT))).flatten())
    img_augmentations_arr.append(np.array(resize(fliplr(img), (IMG_WIDTH, IMG_HEIGHT))).flatten())
    return img_augmentations_arr
def read_image_dataset(dataset_folder, labels, oh_classes_dict, IMG_WIDTH,\
    IMG_HEIGHT, TEST_SET_SIZE, GRAYSCALE, augment=False):
    
    script_dir = os.path.dirname(os.path.abspath("__file__"))
    path = os.path.join(script_dir, dataset_folder)
    data_dir = pathlib.Path(path)

    labels_count = dict()

    x_train = []
    x_test = []
    y_train = []
    y_test = []
    dir_list = None

    for label in labels:
        path = os.path.join(data_dir, label)
        label_code = oh_classes_dict[label]
        dir_list = os.listdir(path)
        dir_list = sorted(dir_list)
        labels_count[label] = len(dir_list)
        count = 0
        for img in tqdm_notebook(dir_list, desc='Loading '+label, position=0, leave=True):
            try:
                # img_array = cv2.imread(os.path.join(path, img), cv2.IMREAD_GRAYSCALE if GRAYSCALE else cv2.IMREAD_COLOR)
                # img_array = cv2.resize(img_array, (IMG_WIDTH, IMG_HEIGHT))
                image = imread(os.path.join(path, img), as_gray=GRAYSCALE)
                img_array = resize(image, (IMG_WIDTH, IMG_HEIGHT))
                img_array = np.array(img_array).flatten()
                if count < labels_count[label] - TEST_SET_SIZE:
                    if augment is False:
                        x_train.append(img_array)
                        y_train.append(label_code)
                    else:
                        img_arrays = get_image_augmentations(image, IMG_WIDTH, IMG_HEIGHT)
                        for augmented_img in img_arrays:
                            x_train.append(augmented_img)
                            y_train.append(label_code)
                        x_train.append(img_array)
                        y_train.append(label_code)                            
                else:
                    x_test.append(img_array)
                    y_test.append(label_code)

                count = count + 1
            except OSError as e:
                print("OSError. Corrupted image, probably!", e, os.path.join(path, img))

    x_train = np.array(x_train)
    x_test = np.array(x_test)
    y_train = np.array(y_train)
    # y_train.reshape(y_train.shape[0], -1)
    y_test = np.array(y_test)
    # y_test.reshape(y_test .shape[0], -1)
    return x_train, y_train, x_test, y_test

def save_np_dataset(np_dataset_folder_name, files_names, np_dataset):
    script_dir = os.path.dirname(os.path.abspath("__file__"))
    path = os.path.join(script_dir, np_dataset_folder_name)
    if not os.path.exists(np_dataset_folder_name):
        os.mkdir(np_dataset_folder_name)

    for file, np_arr in zip(files_names, np_dataset):
        np.save(np_dataset_folder_name+ "/" +file, np_arr)

def load_np_dataset(np_dataset_folder_name, np_data_files):
    data = []
    for file in np_data_files:
        data.append(np.load(np_dataset_folder_name+ "/" +file+".npy"))
    return data[0], data[1], data[2], data[3]

def reshape_np_dataset(x_train, y_train, x_test, y_test, IMG_WIDTH, IMG_HEIGHT):
    x_train = np.array(x_train).reshape(-1, IMG_WIDTH, IMG_WIDTH, 3)
    x_test = np.array(x_test).reshape(-1, IMG_WIDTH, IMG_WIDTH, 3)
    y_train = np.array(y_train)
    y_test = np.array(y_test)
    # Reshape the image data into rows
    # x_train = np.reshape(x_train, (x_train.shape[0], -1))
    # x_test = np.reshape(x_test, (x_test.shape[0], -1))
    print(x_train.shape, x_test.shape)
    return x_train, y_train, x_test, y_test

def scorer_function(y_test_predict, y_test):
    test_size = len(y_test)
    correct_size = np.sum(y_test_predict == y_test)
    accuracy = float(correct_size) / test_size
    # print('Got %d / %d correct => accuracy: %f' %
    #      (correct_size, test_size, accuracy))
    return accuracy

def graph_box(boxes, boxes_names,  x_title,\
    y_title, graphs_folder_name, graph_title, graph_file_name, fig2=None):
    
    if not os.path.exists(graphs_folder_name):
        os.makedirs(graphs_folder_name)

    # Box Graph
    if fig2 is None:
        box_fig= go.Figure()
    else:
        box_fig = fig2

    for score, xi in zip(boxes, boxes_names):
        box_fig.add_trace(go.Box(y=score, name=xi, boxmean=True, showlegend=False))
    box_fig.update_layout(
        title=graph_title,\
        xaxis_title=x_title,\
        yaxis_title=y_title,\
        showlegend=True
    )
    box_fig.write_image(graphs_folder_name + '/' + graph_file_name + '.png')
    box_fig.write_html(graphs_folder_name + '/' + graph_file_name + '.html')

def graph_bar(x,y,
    graph_title,
    graph_file_name,
    x_title,
    y_title,
    graphs_folder_name
    ):
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=x,
        y=y,
        text=y,
        textposition='auto'
    ))
    fig.update_layout(
        title=graph_title,\
        xaxis_title=x_title,\
        yaxis_title=y_title,\
        showlegend=False
    )
    fig.write_image(graphs_folder_name + '/' + graph_file_name + '.png')
    fig.write_html(graphs_folder_name + '/' + graph_file_name + '.html')

# Graphing Cell
def graph_scatter(x, y, x_title, y_title,\
    labels, graphs_folder_name, graph_title, graph_file_name, fig2=None,\
    annotation=None, write=1):

    if not os.path.exists(graphs_folder_name):
        os.makedirs(graphs_folder_name)
    if fig2 is None:
        fig = go.Figure()
    else:
        fig = fig2

    for xi, yi, label in zip(x, y, labels):
        fig.add_trace(go.Scatter(x=xi.get(), y=yi.get(), name=label,
                        mode='lines'))

    fig.update_layout(
        title=graph_title,\
        xaxis_title=x_title,\
        yaxis_title=y_title
    )
    if annotation is not None:
        for ann in annotation:
            fig.add_annotation(
                    x=ann[0],
                    y=ann[1],
                    xref="x",
                    yref="y",
                    text=ann[2],
                    showarrow=True,
                    font=dict(
                        family="Courier New, monospace",
                        size=13,
                        color="#000000"
                        ),
                    align="center",
                    arrowhead=2,
                    arrowsize=1,
                    arrowwidth=2,
                    arrowcolor="#636363",
                    ax=20,
                    ay=-40,
                    bordercolor="#c7c7c7",
                    borderwidth=1,
                    borderpad=3,
                    bgcolor="#ff7f0e",
                    opacity=0.8
                    )
    if write:
        fig.write_image(graphs_folder_name + '\\' + graph_file_name + '.png')
        fig.write_html(graphs_folder_name + '\\' + graph_file_name + '.html')
    return fig

def get_dataset(FIRST_RUN, dataset_folder, labels, oh_classes_dict, IMG_WIDTH, IMG_HEIGHT,\
    TEST_SET_SIZE, GRAYSCALE, np_dataset_folder_name, np_data_files, augment=False):
    if FIRST_RUN == 1:
        [x_train, y_train, x_test, y_test] = read_image_dataset(dataset_folder,\
            labels, oh_classes_dict, IMG_WIDTH, IMG_HEIGHT, TEST_SET_SIZE, GRAYSCALE, augment=augment)
        np_dataset = [x_train, y_train, x_test, y_test]
        save_np_dataset(np_dataset_folder_name, np_data_files, np_dataset)
    else:
        [x_train, y_train, x_test, y_test] = load_np_dataset(np_dataset_folder_name, np_data_files)
    return x_train, y_train, x_test, y_test
# %%

