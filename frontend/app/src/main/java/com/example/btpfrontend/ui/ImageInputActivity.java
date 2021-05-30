package com.example.btpfrontend.ui;

import android.app.Activity;
import android.app.ProgressDialog;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.Toast;

import androidx.annotation.Nullable;
import androidx.appcompat.app.AppCompatActivity;

import com.bumptech.glide.Glide;
import com.example.btpfrontend.R;
import com.example.btpfrontend.model.ResponseModel;
import com.example.btpfrontend.network.ImageRepository;
import com.example.btpfrontend.utils.FileUtils;

import java.io.File;

import io.reactivex.Observer;
import io.reactivex.android.schedulers.AndroidSchedulers;
import io.reactivex.disposables.Disposable;
import io.reactivex.schedulers.Schedulers;
import okhttp3.MediaType;
import okhttp3.MultipartBody;
import okhttp3.RequestBody;

public class ImageInputActivity extends AppCompatActivity {

    private static final int GALLERY_CODE = 100;

    private ImageView inputImage;
    private Button chooseImage, uploadImage;
    private Uri imageUri;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_input);

        inputImage = findViewById(R.id.input_image);
        chooseImage = findViewById(R.id.choose_button);
        uploadImage = findViewById(R.id.upload_button);

        chooseImage.setOnClickListener(v -> {
            Intent intent = new Intent(Intent.ACTION_PICK, android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI);
            startActivityForResult(intent, GALLERY_CODE);
        });

        uploadImage.setOnClickListener(v -> {
            if (imageUri != null) {
                ProgressDialog progressDialog = new ProgressDialog(this);
                progressDialog.setMessage("Uploading Image");
                progressDialog.show();
                File file = FileUtils.getFileFromUri(this, imageUri);
                if (file != null && file.exists()) {
                    RequestBody requestFile = RequestBody.create(MediaType.parse("image/*"), file);
                    MultipartBody.Part imagefile = MultipartBody.Part.createFormData("input", System.currentTimeMillis() + "_" + file.getName(), requestFile);
                    ImageRepository.getInstance().getImageAPI()
                            .uploadImage(imagefile)
                            .subscribeOn(Schedulers.io())
                            .observeOn(AndroidSchedulers.mainThread())
                            .subscribe(new Observer<ResponseModel>() {
                                @Override
                                public void onSubscribe(@io.reactivex.annotations.NonNull Disposable d) {

                                }

                                @Override
                                public void onNext(@io.reactivex.annotations.NonNull ResponseModel response) {
                                    progressDialog.hide();
                                    Intent intent = new Intent(getApplicationContext(), ImageOutputActivity.class);
                                    Bundle bundle = new Bundle();
                                    bundle.putParcelable("response", response);
                                    intent.putExtra("response", bundle);
                                    startActivity(intent);
                                    finish();
                                }

                                @Override
                                public void onError(@io.reactivex.annotations.NonNull Throwable e) {
                                    progressDialog.hide();
                                    Toast.makeText(getApplicationContext(), "Error: " + e.toString(), Toast.LENGTH_LONG).show();
                                    e.printStackTrace();
                                }

                                @Override
                                public void onComplete() {

                                }
                            });
                }
            } else {
                Toast.makeText(this, "Please upload an image.", Toast.LENGTH_LONG).show();
            }
        });
    }

    @Override
    public void onActivityResult(int requestCode, int resultCode, @Nullable Intent data) {
        super.onActivityResult(requestCode, resultCode, data);
        if (requestCode == GALLERY_CODE && resultCode == Activity.RESULT_OK) {
            imageUri = data.getData();
            Glide.with(this)
                    .load(imageUri)
                    .centerCrop()
                    .placeholder(R.drawable.image_icon)
                    .into(inputImage);
        }
    }

    @Override
    public void onBackPressed() {
        startActivity(new Intent(this, MainActivity.class));
        finish();
    }
}