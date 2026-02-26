# Ensure the working directory is correct!
from __init__ import *
import initialsetting as IS
reload(IS)
import imaging as IM
reload(IM)
import training as TR
reload(TR)
import testing as TE
reload(TE)
# from zipfile import ZipFile, ZIP_DEFLATED


if __name__ == '__main__':
    ########################################################################
    # Training
    ########################################################################
    target_dir = os.path.dirname(os.path.abspath(__file__))
    subprocess.run(['python', 'yaml_create.py'])

    yaml_dir = os.path.join(target_dir, 'config.yaml')
    preset = IS.InitialSetting(yaml_dir)

    project_name = 'Train_{}_{}-Infer_{}_{}'.format(
        preset.config.TRAIN.START_DATE,
        preset.config.TRAIN.END_DATE,
        preset.config.INFERENCE.START_DATE,
        preset.config.INFERENCE.END_DATE
    )
    img_data_dir = os.path.join(preset.config.PATHS.IMAGE_DATA_DIR, project_name)

    if os.path.exists(img_data_dir):
        print('Image Data was Already Generated!')
    else:
        genimg = IM.Imaging(setting=preset.config, yaml_dir=yaml_dir)
        genimg.all_Data()

    gpu_label = 0
    cnn_tr = TR.Training(config=preset.config, GPU_label=gpu_label, num_checkpoint=4, delta=0.001)
    # cnn_tr.cnn_training(intra=True, earlystop_mode='intra')
    cnn_tr.cnn_training(intra=True, earlystop_mode='fixed', lr_scheduler='step')

    ########################################################################
    # Testing 
    ########################################################################
    cnn_te = TE.Testing(GPU_label=gpu_label)
    cnn_te.cnn_testing()

    # cnn_tr_all = TE.Testing(data_source='Train_all')
    # cnn_tr_all.cnn_testing()