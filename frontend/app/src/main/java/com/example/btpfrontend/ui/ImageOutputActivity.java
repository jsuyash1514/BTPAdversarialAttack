package com.example.btpfrontend.ui;

import androidx.appcompat.app.AppCompatActivity;

import android.content.Intent;
import android.os.Bundle;
import android.util.Base64;
import android.util.Log;
import android.widget.Button;
import android.widget.ImageView;
import android.widget.TextView;

import com.bumptech.glide.Glide;
import com.example.btpfrontend.R;
import com.example.btpfrontend.model.ResponseModel;
import com.example.btpfrontend.network.BaseRepository;

import org.json.JSONException;
import org.json.JSONObject;

public class ImageOutputActivity extends AppCompatActivity {
    ImageView inputImage, outputImage;
    TextView ind, inc, outd, outc, inlenet, outlenet;
    Button closeButton;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_image_output);
        Bundle bundle = getIntent().getBundleExtra("response");
        ResponseModel responseModel = bundle.getParcelable("response");

        inputImage = findViewById(R.id.input_image);
        outputImage = findViewById(R.id.output_image);
        ind = findViewById(R.id.input_detection);
        inc = findViewById(R.id.input_classification);
        outd = findViewById(R.id.output_detection);
        outc = findViewById(R.id.output_classification);
        inlenet = findViewById(R.id.input_lenet_classification);
        outlenet = findViewById(R.id.output_lenet_classification);
        closeButton = findViewById(R.id.close_button);

        Glide.with(this)
                .load(responseModel.getInputImageUri())
                .centerCrop()
                .placeholder(R.drawable.image_icon)
                .into(inputImage);

        try {
            JSONObject output1 = new JSONObject(responseModel.getOutputjson1());
            JSONObject output2 = new JSONObject(responseModel.getOutputjson2());

            String outputImageurl = BaseRepository.BASE_URL + output1.getString("address");
            Glide.with(this)
                    .load(outputImageurl)
                    .centerCrop()
                    .placeholder(R.drawable.image_icon)
                    .into(outputImage);

            ind.setText(output1.getString("detection_output_normal"));
            inc.setText("Classified as : " + output1.getInt("label_normal"));
            outd.setText(output2.getString("detection_output_adver"));
            outc.setText("Classified as : " + output2.getInt("label_adver"));

            JSONObject inputLenet = new JSONObject(output1.getString("normal_lenet"));
            inlenet.setText("Lenet Model : Classified as " + inputLenet.getInt("lenet_class") + " (" + inputLenet.getDouble("lenet_accuracy") + ")");
            JSONObject outputLenet = new JSONObject(output1.getString("adv_lenet"));
            outlenet.setText("Lenet Model : Classified as " + outputLenet.getInt("lenet_class") + " (" + outputLenet.getDouble("lenet_accuracy") + ")");

        } catch (JSONException e) {
            e.printStackTrace();
        }

        closeButton.setOnClickListener(v -> {
            startActivity(new Intent(this, MainActivity.class));
        });
    }

    @Override
    public void onBackPressed() {
        startActivity(new Intent(this, ImageInputActivity.class));
        finish();
    }
}