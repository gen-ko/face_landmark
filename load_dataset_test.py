from face_landmark import load_dataset

if __name__ == '__main__':
    annotation_root_dir = '/barn2/yuan/datasets/300W_LP_mini/annotations'
    image_root_dir = '/barn2/yuan/datasets/300W_LP_mini/images'
    anno_2ds, anno_3ds, images = load_dataset.load_data(annotation_root_dir, image_root_dir)
    print('anno_2ds shape: ', anno_2ds.shape)
    print('anno_3ds shape: ', anno_3ds.shape)
    print('images shape: ', images.shape)
