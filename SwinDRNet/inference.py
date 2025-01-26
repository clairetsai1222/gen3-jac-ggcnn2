import argparse
import torch
import cv2 as cv
import numpy as np
import PIL.Image as Image
import torch.nn.functional as F

from torchvision import transforms
from scipy.ndimage import zoom
from networks.SwinDRNet import SwinDRNet
from config import get_config

class SwinDRNetPipeline():
    def __init__(self, model_path):
        self.args = self.parser_init()
        self.config = get_config(self.args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = SwinDRNet(self.config, img_size=self.args.img_size, num_classes=self.args.num_classes).cuda()
        msg = self.model.load_state_dict(torch.load(model_path)['model_state_dict'])
        print("self trained swin unet", msg)
        self.model.eval()

    def inference(self, rgb: np.ndarray, depth: np.ndarray, target_size_factor = 1, depth_zoom_factor = 1000):
        '''
        rgb should be in RGB and is np.array;
        depth should be in one channel(mm) and is np.array.
        target size factor can be changed for better result.
        depth zoom factor can adjust the value of distance for better result. 
        '''
        #target_size = (target_size_factor*self.args.img_size, target_size_factor*self.args.img_size)
        rgb_zoom = (target_size_factor*self.args.img_size/rgb.shape[0], target_size_factor*self.args.img_size/rgb.shape[1],1)
        depth_zoom = (target_size_factor*self.args.img_size/depth.shape[0], target_size_factor*self.args.img_size/depth.shape[1])
        h,w,_ = rgb.shape
        
        # preprocess RGB
        _rgb = rgb.astype(np.float32)/255
        #_rgb = cv.resize(rgb, target_size ,interpolation=cv.INTER_NEAREST)
        _rgb = zoom(_rgb,rgb_zoom).astype(np.float32)
        _rgb = transforms.ToTensor()(_rgb)
        _rgb = _rgb.unsqueeze(0)
        # print(_rgb)
        #transforms.ToPILImage()(_rgb.squeeze(0)).show()

        # preprocess depth
        _depth = depth.astype(np.float32)/depth_zoom_factor
        _depth = zoom(_depth,depth_zoom).astype(np.float32)
        #_depth = cv.resize(_depth, target_size ,interpolation=cv.INTER_NEAREST)
        _depth = _depth[np.newaxis, ...] 
        _depth[_depth <= 0] = 0.0
        _depth = _depth.squeeze(0)
        _depth = transforms.ToTensor()(_depth)
        _depth = _depth.unsqueeze(0)
        #transforms.ToPILImage()(_depth.squeeze(0)).show()
        # to device
        _rgb = _rgb.to(self.device)
        _depth = _depth.to(self.device)
        
        #print('='*20)
        # print(_depth)
        #Image.fromarray(np.array(_rgb.cpu()).squeeze(), 'RGB').show()

        with torch.no_grad():  
            pred_ds, pred_ds_initial, confidence_sim_ds, confidence_initial = self.model(_rgb, _depth)
            if  pred_ds.shape[2:] != (h,w):
                # upsampling to origin rgb's resolution
                pred_ds = F.interpolate(pred_ds,(h,w),mode='bilinear')
            output_depth = np.array(pred_ds.cpu()).squeeze(0).squeeze(0)*depth_zoom_factor
        # confidence map
        confidence_map_ini = np.array(confidence_initial.cpu()).squeeze(0).squeeze(0)
        confidence_map_ds = np.array(confidence_sim_ds.cpu()).squeeze(0).squeeze(0)
        # output_size = (output_depth.shape[1], output_depth.shape[0])
        ini_zoom = (output_depth.shape[0]/confidence_map_ini.shape[0], output_depth.shape[1]/confidence_map_ini.shape[1])
        ds_zoom = (depth.shape[0]/confidence_map_ds.shape[0], depth.shape[1]/confidence_map_ds.shape[1])
        output_depth_mapped = output_depth * zoom(confidence_map_ini, ini_zoom)\
                              + depth * zoom(confidence_map_ds, ds_zoom)
        # output_depth_mapped = output_depth * cv.resize(np.array(confidence_initial.cpu()).squeeze(0).squeeze(0), output_size)\
        #                       + depth * cv.resize(np.array(confidence_sim_ds.cpu()).squeeze(0).squeeze(0), output_size)
        #return cv.resize(np.array(confidence_initial.cpu()).squeeze(0).squeeze(0), output_size)*2550
        #return output_depth
        return output_depth_mapped
    
    def parser_init(self):
        parser = argparse.ArgumentParser()
        parser.add_argument('--mask_transparent', action='store_true', default=True, help='material mask')
        parser.add_argument('--mask_specular', action='store_true', default=True, help='material mask')
        parser.add_argument('--mask_diffuse', action='store_true', default=True, help='material mask')
        parser.add_argument('--train_data_path', type=str,
                            default='/data/DREDS/DREDS-CatKnown/train', help='root dir for training dataset')
        parser.add_argument('--val_data_path', type=str,
                            default='/data/DREDS/DREDS-CatKnown/test', help='root dir for validation dataset')
        parser.add_argument('--val_data_type', type=str,
                            default='sim', help='type of val dataset (real/sim)')
        parser.add_argument('--output_dir', type=str, 
                            default='results/DREDS_CatKnown', help='output dir')
        parser.add_argument('--checkpoint_save_path', type=str, 
                            default='models/DREDS', help='Choose a path to save checkpoints')
        parser.add_argument('--decode_mode', type=str, 
                            default='multi_head', help='Select encode mode')
        parser.add_argument('--val_interation_interval', type=int, 
                            default=5000, help='The iteration interval to perform validation')
        parser.add_argument('--percentageDataForTraining', type=float, 
                            default=1.0, help='The percentage of full training data for training')
        parser.add_argument('--percentageDataForVal', type=float, 
                            default=1.0, help='The percentage of full training data for training')
        parser.add_argument('--num_classes', type=int,
                            default=9, help='output channel of network')
        parser.add_argument('--max_epochs', type=int, default=20,
                            help='maximum epoch number to train')
        parser.add_argument('--batch_size', type=int, default=64,
                            help='batch_size per gpu')
        parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
        parser.add_argument('--deterministic', type=int,  default=1,
                            help='whether use deterministic training')
        parser.add_argument('--base_lr', type=float,  default=0.0001,
                            help='segmentation network learning rate')
        parser.add_argument('--img_size', type=int,
                            default=224, help='input patch size of network input')
        parser.add_argument('--seed', type=int,
                            default=1234, help='random seed')
        parser.add_argument('--cfg', type=str, default="configs/swin_tiny_patch4_window7_224_lite.yaml", metavar="FILE", help='path to config file', )
        parser.add_argument(
                            "--opts",
                            help="Modify config options by adding 'KEY VALUE' pairs. ",
                            default=None,
                            nargs='+',
                            )
        parser.add_argument('--zip', action='store_true', default=True, help='use zipped dataset instead of folder dataset')
        parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                            help='no: no cache, '
                            'full: cache all data, '
                            'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
        parser.add_argument('--resume',type=str, default='./output-1/epoch_149.pth', help='resume from checkpoint')
        parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
        parser.add_argument('--use-checkpoint', action='store_true',
                            help="whether to use gradient checkpointing to save memory")
        parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                            help='mixed precision opt level, if O0, no amp is used')
        parser.add_argument('--tag', help='tag of experiment')
        parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
        parser.add_argument('--throughput', action='store_true', help='Test throughput only')
        args = args = parser.parse_args()
        
        return args