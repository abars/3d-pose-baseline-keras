#insert this code to train function of predict_3dpose.py
#  export_dataset(train_set_2d,data_mean_2d,data_std_2d,train_set_3d,data_mean_3d,data_std_3d,"train.h5")
#  export_dataset(test_set_2d,data_mean_2d,data_std_2d,test_set_3d,data_mean_3d,data_std_3d,"test.h5")

def export_dataset(train_set_2d,data_mean_2d,data_std_2d,train_set_3d,data_mean_3d,data_std_3d,hdf5_name):
  # Figure out how many frames we have
  data_x=train_set_2d
  data_y=train_set_3d
  camera_frame=FLAGS.camera_frame

  n = 0
  for key2d in data_x.keys():
    n2d, _ = data_x[ key2d ].shape
    n = n + n2d

  encoder_inputs  = np.zeros((n, 32), dtype=float)
  decoder_outputs = np.zeros((n, 48), dtype=float)

  # Put all the data into big arrays
  idx = 0
  for key2d in data_x.keys():
    (subj, b, fname) = key2d
    # keys should be the same if 3d is in camera coordinates
    key3d = key2d if (camera_frame) else (subj, b, '{0}.h5'.format(fname.split('.')[0]))
    key3d = (subj, b, fname[:-3]) if fname.endswith('-sh') and camera_frame else key3d

    n2d, _ = data_x[ key2d ].shape
    encoder_inputs[idx:idx+n2d, :]  = data_x[ key2d ]
    decoder_outputs[idx:idx+n2d, :] = data_y[ key3d ]
    idx = idx + n2d

  # Write DataSet
  print("Export data to "+hdf5_name)
  print(encoder_inputs)
  print(decoder_outputs)
  print(encoder_inputs.shape)
  print(decoder_outputs.shape)
  print(data_mean_2d.shape)
  print(data_std_2d.shape)
  print(data_mean_3d.shape)
  print(data_std_3d.shape)

  with h5py.File(hdf5_name, 'w') as f:
    f.create_dataset('encoder_inputs', data=encoder_inputs)
    f.create_dataset('decoder_outputs', data=decoder_outputs)
    f.create_dataset('data_mean_2d', data=data_mean_2d)
    f.create_dataset('data_std_2d', data=data_std_2d)
    f.create_dataset('data_mean_3d', data=data_mean_3d)
    f.create_dataset('data_std_3d', data=data_std_3d)
