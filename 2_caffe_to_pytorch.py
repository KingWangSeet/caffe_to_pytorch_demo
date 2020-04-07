from ssd.config import cfg
from ssd.modeling.detector import build_detection_model
import numpy as np
import torch
torch.set_default_tensor_type('torch.FloatTensor')

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'

#refindet detection model
model = build_detection_model(cfg)

new_weight_dic = model.state_dict().copy()

var_to_weights_map = np.load('refinedet_res10_2cls_fromcaffe_origin_ding_2cls_v7_addp6_up.npy',encoding='latin1',allow_pickle=True)[()]


#SEbasicBlock caffe->pytorch
def extract_from_seres10_sebasicblock(new_weight_dic, var_to_weights_map, prefix, stage):
    block =0 
    #var_to_weights_map key
    conv_name_base = 'layer'+str(stage-1)+'.'+str(block) #layer1.0
    bn_name_base = 'layer' + str(stage-1)+'.'+str(block)
    
    caffe_name_base = 'conv' + str(stage)+'_1' #conv2-1
    caffe_name_base_x1 = 'conv' + str(stage)+'_1/x1' #conv2_1/x1
    caffe_name_base_x2 = 'conv' + str(stage)+'_1/x2' #conv2_1/x2
       
    #SE basicblok conv x1 x2
    name = [[conv_name_base+'.conv1', bn_name_base+'.bn1', caffe_name_base_x1], #layer1.0.conv1  layer1.0.bn1 conv2_1/x1
            [conv_name_base+'.conv2', bn_name_base+'.bn2', caffe_name_base_x2]]#layer1.0.conv2 layer1.0.bn2 onv2_1/x2
    
    for conv_name, bn_name, caffe_name in name:         
        # conv  conv2_1/x1.weight -->layer1.0.conv1.weight
        print(conv_name, bn_name, caffe_name)
        #layer1.0.conv1 layer1.0.bn1 conv2_1/x1
        weights = var_to_weights_map[caffe_name+'.weight']
#         print(weights.shape)
        new_weight_dic[prefix+ conv_name +'.weight'] = weights #pytorch
#         print(new_weight_dic[prefix+ conv_name +'.weight'].shape)

        # conv2_1/x1/scale(gamma,beta)->layer1.0.bn1(weight,bias)
        gamma = var_to_weights_map[caffe_name+'/scale.weight']
        new_weight_dic[prefix+ bn_name +'.weight'] = gamma 
        beta = var_to_weights_map[caffe_name+'/scale.bias']
        new_weight_dic[prefix+ bn_name +'.bias'] = beta 
        
        # conv2_1/x1/bn(weight,bias)->layer1.0.bn1(mean,var)
        mean = var_to_weights_map[caffe_name+'/bn.weight']
        new_weight_dic[prefix+ bn_name+'.running_mean'] = mean

        variance = var_to_weights_map[caffe_name+'/bn.bias']
        new_weight_dic[prefix+ bn_name+'.running_var'] = variance

    conv_se_fc_name_0 = conv_name_base + '.se.fc.0' #layer1.0.se.fc.0
    conv_se_fc_name_2 = conv_name_base + '.se.fc.2' #layer1.0.se.fc.2
    
    caffe_se_fc_name_0 = 'fc'+str(stage)+'_1/sqz' #fc2_1/sqz
    caffe_se_fc_name_2 = 'fc'+str(stage)+'_1/exc' #fc2_1/exc
    
    fc2_1_sqz_weight = var_to_weights_map[caffe_se_fc_name_0+'.weight']
    fc2_1_sqz_bias  = var_to_weights_map[caffe_se_fc_name_0+'.bias']

    fc2_1_exc_weight = var_to_weights_map[caffe_se_fc_name_2+'.weight']
    fc2_1_exc_bias  = var_to_weights_map[caffe_se_fc_name_2+'.bias']
    
    new_weight_dic[prefix+ conv_se_fc_name_0+'.weight'] = fc2_1_sqz_weight
    new_weight_dic[prefix+ conv_se_fc_name_0+'.bias'] = fc2_1_sqz_bias
    
    new_weight_dic[prefix+ conv_se_fc_name_2+'.weight'] = fc2_1_exc_weight
    new_weight_dic[prefix+ conv_se_fc_name_2+'.bias'] = fc2_1_exc_bias   

    caffe_short_cut_name = caffe_name_base + '/prj' #conv3_1/prj
    print(caffe_short_cut_name)
    conv_name = conv_name_base + '.downsample' #layer2.0.downsample
    bn_name = bn_name_base + '.downsample' #layer2.0.downsample
    # conv
    weights = var_to_weights_map[caffe_short_cut_name+'.weight']
    new_weight_dic[prefix+ conv_name+'.0.weight'] = weights
    # bn
    mean = var_to_weights_map[caffe_short_cut_name+'/bn.weight']
    new_weight_dic[prefix+ bn_name+'.1.running_mean'] = mean
    variance = var_to_weights_map[caffe_short_cut_name+'/bn.bias']
    new_weight_dic[prefix+ bn_name+'.1.running_var'] = variance

    gamma = var_to_weights_map[caffe_short_cut_name+'/scale.weight']
    new_weight_dic[prefix+ bn_name+'.1.weight'] = gamma  
    beta = var_to_weights_map[caffe_short_cut_name+'/scale.bias']
    new_weight_dic[prefix+ bn_name+'.1.bias'] = beta   
     
    print('fininsing loading weight from ' +caffe_name_base + ' to  '+ conv_name_base)
        
def caffe2torch(var_to_weights_map, new_weight_dic):    
    # Resnet
    prefix = 'backbone.'
    # Stage 1
    # conv1.weights
#     conv1.weight (64, 3, 7, 7)
#     conv1/bn.weight (64,)
#     conv1/bn.bias (64,)
#     conv1/scale.weight (64,)
#     conv1/scale.bias (64,)
    weights = var_to_weights_map['conv1.weight'] #caffe layer name
    new_weight_dic[prefix+'conv1.weight'] = weights #pytorch
    
    #bn1.weight
    gamma = var_to_weights_map['conv1/scale.weight']
    new_weight_dic[prefix+'bn1.weight'] = gamma
    
    #bn1.bias
    beta = var_to_weights_map['conv1/scale.bias']
    new_weight_dic[prefix+'bn1.bias'] = beta
    
    #bn1.mean
    mean = var_to_weights_map['conv1/bn.weight']
    new_weight_dic[prefix+'bn1.running_mean'] = mean
    
    # bn1.variance
    variance = var_to_weights_map['conv1/bn.bias']
    new_weight_dic[prefix+'bn1.running_var'] = variance
    
#     print(new_weight_dic)
#     exit()
    # Stage 2  3 4 5 
    for i in range(2,6):
        extract_from_seres10_sebasicblock(new_weight_dic, var_to_weights_map, prefix, stage=i)
#     exit()
    ##-------C5 _lateral TL6_1 TL6_2 P6----------------
    #c5_lateral.0.weight
    #c5_lateral.0.bias
    #c5_lateral.2.weight
    #c5_lateral.2.bias
    #c5_lateral.4.weight
    #c5_lateral.4.bias
    
    #TL6_1.weight
    # TL6_1.bias
    # TL6_2.weight
    # TL6_2.bias
    # P6.weight
    # P6.bias
    
    TL6_1_weight = var_to_weights_map['TL6_1.weight']
    TL6_1_bias = var_to_weights_map['TL6_1.bias']
    
    TL6_2_weight = var_to_weights_map['TL6_2.weight']
    TL6_2_bias = var_to_weights_map['TL6_2.bias']
    
    P6_weight = var_to_weights_map['P6.weight']
    P6_bias = var_to_weights_map['P6.bias']
    
    new_weight_dic[prefix+'c5_lateral.0.weight'] = TL6_1_weight
    new_weight_dic[prefix+'c5_lateral.0.bias'] = TL6_1_bias
    
    new_weight_dic[prefix+'c5_lateral.2.weight'] = TL6_2_weight
    new_weight_dic[prefix+'c5_lateral.2.bias'] = TL6_2_bias
    
    new_weight_dic[prefix+'c5_lateral.4.weight'] = P6_weight
    new_weight_dic[prefix+'c5_lateral.4.bias'] = P6_bias
    
    #C4_lateral  TL5_1 TL5_2
    #c4_lateral.0.weight
    #c4_lateral.0.bias
    #c4_lateral.2.weight
    #c4_lateral.2.bias
    
    # TL5_1.weight
    # TL5_1.bias
    # TL5_2.weight
    # TL5_2.bias
    TL5_1_weight = var_to_weights_map['TL5_1.weight']
    TL5_1_bias  = var_to_weights_map['TL5_1.bias']
    
    TL5_2_weight = var_to_weights_map['TL5_2.weight']
    TL5_2_bias  = var_to_weights_map['TL5_2.bias']
    
    new_weight_dic[prefix+'c4_lateral.0.weight'] = TL5_1_weight
    new_weight_dic[prefix+'c4_lateral.0.bias'] = TL5_1_bias
    new_weight_dic[prefix+'c4_lateral.2.weight'] = TL5_2_weight
    new_weight_dic[prefix+'c4_lateral.2.bias'] = TL5_2_bias
    
    #p4_conv P5
    #p4_conv.0.weight
    #p4_conv.0.bias
    
    # P5.weight
    # P5.bias
    p4_conv_weight = var_to_weights_map['P5.weight']
    p4_conv_bias = var_to_weights_map['P5.bias']
    
    new_weight_dic[prefix+'p4_conv.0.weight'] = p4_conv_weight
    new_weight_dic[prefix+'p4_conv.0.bias'] = p4_conv_bias
    
    # P6-up.weight
    # p6-up.bias
    p6_up_weight = var_to_weights_map['P6-up.weight']
    p6_up_bias = var_to_weights_map['P6-up.bias']
    
    new_weight_dic[prefix+'p6_up.weight'] = p6_up_weight
    new_weight_dic[prefix+'p6_up.bias'] = p6_up_bias
    
    
    ###ARM outout layer
    ### c4_arm_loc_layer<-  block_4_1_mbox_loc
    ### c4_arm_cls_layer<-  block_4_1_mbox_conf
    ### c5_arm_loc_layer<-  block_5_1_mbox_loc 
    ### c5_arm_cls_layer<-  block_5_1_mbox_conf
    
    c4_arm_loc_layer_weight = var_to_weights_map['block_4_1_mbox_loc.weight']
    c4_arm_loc_layer_bias  = var_to_weights_map['block_4_1_mbox_loc.bias']
    new_weight_dic[prefix+'c4_arm_loc_layer.weight'] = c4_arm_loc_layer_weight
    new_weight_dic[prefix+'c4_arm_loc_layer.bias'] = c4_arm_loc_layer_bias
    
    c4_arm_cls_layer_weight = var_to_weights_map['block_4_1_mbox_conf.weight']
    c4_arm_cls_layer_bias  = var_to_weights_map['block_4_1_mbox_conf.bias']
    new_weight_dic[prefix+'c4_arm_cls_layer.weight'] = c4_arm_cls_layer_weight
    new_weight_dic[prefix+'c4_arm_cls_layer.bias'] = c4_arm_cls_layer_bias
    
    c5_arm_loc_layer_weight = var_to_weights_map['block_5_1_mbox_loc.weight']
    c5_arm_loc_layer_bias  = var_to_weights_map['block_5_1_mbox_loc.bias']
    new_weight_dic[prefix+'c5_arm_loc_layer.weight'] = c5_arm_loc_layer_weight
    new_weight_dic[prefix+'c5_arm_loc_layer.bias'] = c5_arm_loc_layer_bias
    
    c5_arm_cls_layer_weight = var_to_weights_map['block_5_1_mbox_conf.weight']
    c5_arm_cls_layer_bias  = var_to_weights_map['block_5_1_mbox_conf.bias']
    new_weight_dic[prefix+'c5_arm_cls_layer.weight'] = c5_arm_cls_layer_weight
    new_weight_dic[prefix+'c5_arm_cls_layer.bias'] = c5_arm_cls_layer_bias
    
    ###ODM outout layer(reverse)
    ### p4_odm_loc_layer<- P5_mbox_loc
    ### p4_odm_cls_layer<-  P5_mbox_conf1
    ### p5_odm_loc_layer<- P6_mbox_loc
    ### p5_odm_cls_layer<-  P6_mbox_conf1
    
    p4_odm_loc_layer_weight = var_to_weights_map['P5_mbox_loc.weight']
    p4_odm_loc_layer_bias  = var_to_weights_map['P5_mbox_loc.bias']
    new_weight_dic[prefix+'p4_odm_loc_layer.weight'] = p4_odm_loc_layer_weight
    new_weight_dic[prefix+'p4_odm_loc_layer.bias'] = p4_odm_loc_layer_bias

    ### deploy_res10_finetune_6c_origin.prototxt
#     p4_odm_cls_layer_weight = var_to_weights_map['P5_mbox_conf1.weight']
#     p4_odm_cls_layer_bias  = var_to_weights_map['P5_mbox_conf1.bias']
    
    ### test.prototxt
    p4_odm_cls_layer_weight = var_to_weights_map['P5_mbox_conf.weight']
    p4_odm_cls_layer_bias  = var_to_weights_map['P5_mbox_conf.bias']
    
    new_weight_dic[prefix+'p4_odm_cls_layer.weight'] = p4_odm_cls_layer_weight
    new_weight_dic[prefix+'p4_odm_cls_layer.bias'] = p4_odm_cls_layer_bias
    
    p5_odm_loc_layer_weight = var_to_weights_map['P6_mbox_loc.weight']
    p5_odm_loc_layer_bias  = var_to_weights_map['P6_mbox_loc.bias']
    new_weight_dic[prefix+'p5_odm_loc_layer.weight'] = p5_odm_loc_layer_weight
    new_weight_dic[prefix+'p5_odm_loc_layer.bias'] = p5_odm_loc_layer_bias
 
    ### deploy_res10_finetune_6c_origin.prototxt
#     p5_odm_cls_layer_weight = var_to_weights_map['P6_mbox_conf1.weight']
#     p5_odm_cls_layer_bias  = var_to_weights_map['P6_mbox_conf1.bias']
    
#     #### test.prototxt
    p5_odm_cls_layer_weight = var_to_weights_map['P6_mbox_conf.weight']
    p5_odm_cls_layer_bias  = var_to_weights_map['P6_mbox_conf.bias']
    
    new_weight_dic[prefix+'p5_odm_cls_layer.weight'] = p5_odm_cls_layer_weight
    new_weight_dic[prefix+'p5_odm_cls_layer.bias'] = p5_odm_cls_layer_bias
#     print(p5_odm_cls_layer_bias)
#     exit()
    
caffe2torch(var_to_weights_map, new_weight_dic)
new_weight_dic_tensor={}
for key ,value in new_weight_dic.items():
    new_weight_dic_tensor[key]= torch.tensor(value).to(torch.float32)
# for key in model.state_dict().keys():
#     print(key) #116  
    
print(len(new_weight_dic_tensor)) #116    
model.load_state_dict(new_weight_dic_tensor)
torch.save(model.state_dict(), 'pretrain_refinedet10_2cls_caffe_params_v2_addp6.pth')
print('saved done.')