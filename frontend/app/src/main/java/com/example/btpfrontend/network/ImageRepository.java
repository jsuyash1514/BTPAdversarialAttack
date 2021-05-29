package com.example.btpfrontend.network;

public class ImageRepository {
    private static ImageRepository imageRepository;
    private ImageAPI imageAPI;

    public ImageRepository() {
        imageAPI = BaseRepository
                .getDefaultBuilder()
                .build()
                .create(ImageAPI.class);
    }

    public static synchronized ImageRepository getInstance() {
        if (imageRepository == null) {
            imageRepository = new ImageRepository();
        }
        return imageRepository;
    }

    public ImageAPI getImageAPI() {
        return imageAPI;
    }
}
