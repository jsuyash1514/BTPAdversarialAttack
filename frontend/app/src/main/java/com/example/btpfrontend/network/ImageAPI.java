package com.example.btpfrontend.network;

import com.example.btpfrontend.model.ResponseModel;

import io.reactivex.Observable;
import okhttp3.MultipartBody;
import okhttp3.ResponseBody;
import retrofit2.http.Multipart;
import retrofit2.http.POST;
import retrofit2.http.Part;

public interface ImageAPI {
    @Multipart
    @POST("/result/")
    Observable<ResponseModel> uploadImage(@Part MultipartBody.Part image);
}
