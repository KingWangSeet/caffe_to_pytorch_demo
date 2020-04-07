####--------------------Get pytorch refinedet_res10 model from caffe model ---------------------------#############
####--------save the caffe model to numpy .npy (key->value)
####------------------------------------------------------------------------------------------##############

import sys
sys.path.insert(0, 'caffe/python')
import sys
import os
import caffe
import numpy as np

###input 
cf_prototxt = "test.prototxt" #input the refindedet res10cfg from ding caffe model 2cl
cf_model = "refindet512_seres10.caffemodel" #input caffe_model from ding caffe model

checkpoint_path = "refinedet_res10_2cls_fromcaffe.npy"  #output caffe weights

net = caffe.Net(cf_prototxt, cf_model, caffe.TEST)
p = []
conv_layer_nobias={'conv1','conv2_1/prj','conv2_1/x1','conv2_1/x2','conv3_1/x1','conv3_1/x2','conv3_1/prj','conv4_1/x1','conv4_1/x2','conv4_1/prj','conv5_1/x1','conv5_1/x2','conv5_1/prj'}

weight_dict={}
##Extraxt the weight from caffe model and save to the weight_dict for net step 
for param_name, param_values in net.params.items():
    
    if param_name in conv_layer_nobias : 
        print(param_name+'------>'+str(len(param_values)))
        print(param_name+'.weight' + ' '+str(param_values[0].data.shape))
#         print(param_name+'.weight' + ' '+str(param_values[0].data))
        weight = param_values[0].data
        weight_dict[param_name+'.weight'] = weight
        p.append(weight)
    else:
        if 'bn' not in param_name:
            print(param_name+'------>'+str(len(param_values)))
            print(param_name+'.weight'+' '+str(param_values[0].data.shape))
            print(param_name+'.bias'+' '+str(param_values[1].data.shape))


            weight = param_values[0].data
            bias = param_values[1].data
            weight_dict[param_name+'.weight'] = weight
            weight_dict[param_name+'.bias'] = bias
            p.append(weight)
            p.append(bias)
        else: # inlcude bn layer
            print(param_name+'------>'+str(len(param_values)))
            print('bn laer--->'+param_name)
            print(param_name+'.weight'+' '+str(param_values[0].data.shape))
            print(param_name+'.bias'+' '+str(param_values[1].data.shape))
            print(param_name+'.third'+' '+str(param_values[2].data))


            weight = param_values[0].data/param_values[2].data
            bias = param_values[1].data/param_values[2].data
            weight_dict[param_name+'.weight'] = weight
            weight_dict[param_name+'.bias'] = bias
            p.append(weight)
            p.append(bias)    
# exit()
        
# print(len(weight_dict) #106
np.save(checkpoint_path,weight_dict)
print('save the caffemodel weights to the '+checkpoint_path + ' npy file')

