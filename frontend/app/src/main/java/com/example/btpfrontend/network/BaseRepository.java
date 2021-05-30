package com.example.btpfrontend.network;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.util.concurrent.TimeUnit;

import okhttp3.OkHttpClient;
import retrofit2.Retrofit;
import retrofit2.adapter.rxjava2.RxJava2CallAdapterFactory;
import retrofit2.converter.gson.GsonConverterFactory;

public class BaseRepository {
    public static final int DEFAULT_TIMEOUT = 300;

    public static final String BASE_URL = "http://192.168.37.250:8000/";

    public static Retrofit.Builder getDefaultBuilder() {
        OkHttpClient.Builder okHttpClientBuilder = new OkHttpClient.Builder()
                .readTimeout(DEFAULT_TIMEOUT, TimeUnit.SECONDS)
                .connectTimeout(30, TimeUnit.SECONDS);

        Retrofit.Builder builder = new Retrofit.Builder()
                .baseUrl(BASE_URL)
                .client(okHttpClientBuilder.build())
                .addConverterFactory(GsonConverterFactory.create(getDefaultGson()))
                .addCallAdapterFactory(RxJava2CallAdapterFactory.create());

        return builder;
    }

    private static Gson getDefaultGson() {
        return new GsonBuilder()
                .setDateFormat("yyyy-MM-dd'T'HH:mm:ss'Z'")
                .create();
    }
}
