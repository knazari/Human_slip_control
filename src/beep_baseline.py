import glob
import time
import numpy as np
import pandas as pd
import simpleaudio as sa

wave_obj = sa.WaveObject.from_wave_file('/home/kia/catkin_ws/src/data_collection_human_test/assets/beep-07a.wav')

subject_data_path = "/home/kia/catkin_ws/src/data_collection_human_test/data/subject_004/baseline"

files = sorted(glob.glob(subject_data_path + '/*'))[-1]

meta_data = pd.read_csv(files + '/meta_data.csv')
task_duration = meta_data['task_completion_time']
print(task_duration)

time.sleep(1)
play_obj = wave_obj.play()
time.sleep(task_duration[0]/2.)
play_obj = wave_obj.play()
time.sleep(task_duration[0]/2.)
play_obj = wave_obj.play()
play_obj.wait_done()


time.sleep(3)
play_obj = wave_obj.play()
time.sleep(0.20)
play_obj = wave_obj.play()
time.sleep(0.20)
play_obj = wave_obj.play()
play_obj.wait_done()

