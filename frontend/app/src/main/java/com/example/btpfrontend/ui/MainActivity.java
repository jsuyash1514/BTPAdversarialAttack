package com.example.btpfrontend.ui;

import android.content.Intent;
import android.os.Bundle;
import android.widget.Button;

import androidx.appcompat.app.AppCompatActivity;

import com.example.btpfrontend.R;

public class MainActivity extends AppCompatActivity {

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        Button start = findViewById(R.id.start_button);
        start.setOnClickListener(v -> {
            startActivity(new Intent(this, ImageInputActivity.class));
            finish();
        });
    }
}