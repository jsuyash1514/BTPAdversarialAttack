package com.example.btpfrontend.model;

import android.os.Parcel;
import android.os.Parcelable;

import com.google.gson.annotations.SerializedName;

public class ResponseModel implements Parcelable {

    @SerializedName("input")
    String inputImageUri;

    @SerializedName("output1")
    String outputjson1;

    @SerializedName("output2")
    String outputjson2;

    private ResponseModel(Parcel in) {
        inputImageUri = in.readString();
        outputjson1 = in.readString();
        outputjson2 = in.readString();
    }

    public static final Parcelable.Creator<ResponseModel> CREATOR
            = new Parcelable.Creator<ResponseModel>() {
        public ResponseModel createFromParcel(Parcel in) {
            return new ResponseModel(in);
        }

        public ResponseModel[] newArray(int size) {
            return new ResponseModel[size];
        }
    };

    public String getInputImageUri() {
        return inputImageUri;
    }

    public void setInputImageUri(String inputImageUri) {
        this.inputImageUri = inputImageUri;
    }

    public String getOutputjson1() {
        return outputjson1;
    }

    public void setOutputjson1(String outputjson1) {
        this.outputjson1 = outputjson1;
    }

    public String getOutputjson2() {
        return outputjson2;
    }

    public void setOutputjson2(String outputjson2) {
        this.outputjson2 = outputjson2;
    }

    @Override
    public int describeContents() {
        return 0;
    }

    @Override
    public void writeToParcel(Parcel dest, int flags) {
        dest.writeString(inputImageUri);
        dest.writeString(outputjson1);
        dest.writeString(outputjson2);
    }
}
