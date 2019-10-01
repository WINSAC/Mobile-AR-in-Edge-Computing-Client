/*
 * Copyright 2019 The TensorFlow Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *       http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.tensorflow.lite.examples.detection;

import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Paint.Style;
import android.graphics.RectF;
import android.graphics.Typeface;
import android.media.Image;
import android.media.ImageReader;
import android.media.ImageReader.OnImageAvailableListener;
import android.os.Environment;
import android.os.SystemClock;
import android.os.Trace;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.widget.Toast;

import java.io.File;
import java.io.BufferedWriter;
import java.io.FileWriter;
import java.io.DataOutputStream;
import java.io.IOException;
import java.net.InetAddress;
import java.net.Socket;
import java.net.UnknownHostException;
import java.util.LinkedList;
import java.util.List;
import org.tensorflow.lite.examples.detection.customview.OverlayView;
import org.tensorflow.lite.examples.detection.customview.OverlayView.DrawCallback;
import org.tensorflow.lite.examples.detection.env.BorderedText;
import org.tensorflow.lite.examples.detection.env.ImageUtils;
import org.tensorflow.lite.examples.detection.env.Logger;
import org.tensorflow.lite.examples.detection.tflite.Classifier;
import org.tensorflow.lite.examples.detection.tflite.TFLiteObjectDetectionAPIModel;
import org.tensorflow.lite.examples.detection.tracking.MultiBoxTracker;

/**
 * An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
 * objects.
 */
public class DetectorActivity extends CameraActivity implements OnImageAvailableListener {
  private static final Logger LOGGER = new Logger();

  // Configuration values for the YOLOv3 model.
  private static final int TF_OD_API_INPUT_SIZE = 608;
  private static final boolean TF_OD_API_IS_QUANTIZED = true;
  private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
  private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
  private static final DetectorMode MODE = DetectorMode.TF_OD_API;
  // Minimum detection confidence to track a detection.
  private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;
  private static final boolean MAINTAIN_ASPECT = false;
  //private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);
  private static final Size DESIRED_PREVIEW_SIZE = new Size(800, 600);
  private static final boolean SAVE_PREVIEW_BITMAP = false;
  private static final float TEXT_SIZE_DIP = 10;
  OverlayView trackingOverlay;
  private Integer sensorOrientation;

  private Classifier detector;

  private long lastProcessingTimeMs;
  private Bitmap rgbFrameBitmap = null;
  private Bitmap croppedBitmap = null;
  private Bitmap cropCopyBitmap = null;

  private boolean computingDetection = false;

  private long timestamp = 0;

  private Matrix frameToCropTransform;
  private Matrix cropToFrameTransform;

  private MultiBoxTracker tracker;

  private byte[] luminanceCopy;

  private BorderedText borderedText;

  /*---------------Change 2--------------------*/
  private boolean SAVE_PREVIEW = true;
  private boolean isProcessingFrame = false;
  private int[] rgbBytes = null;
  private byte[][] yuvBytes = new byte[3][];
  private int yRowStride;
  private Runnable imageConverter;
  private Runnable postInferenceCallback;
  private int frameID = 0;
  private boolean feedback = false;
  public static final String TAA = "Latency";
  //private boolean offloading = false;
  /*---------------end 2--------------------*/

  @Override
  public void onPreviewSizeChosen(final Size size, final int rotation) {
    Log.i(TAG,"03");
    final float textSizePx =
        TypedValue.applyDimension(
            TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
    borderedText = new BorderedText(textSizePx);
    borderedText.setTypeface(Typeface.MONOSPACE);

    tracker = new MultiBoxTracker(this);

    int cropSize = TF_OD_API_INPUT_SIZE;

    try {
      detector =
          TFLiteObjectDetectionAPIModel.create(
              getAssets(),
              TF_OD_API_MODEL_FILE,
              TF_OD_API_LABELS_FILE,
              TF_OD_API_INPUT_SIZE,
              TF_OD_API_IS_QUANTIZED);
      cropSize = TF_OD_API_INPUT_SIZE;
    } catch (final IOException e) {
      e.printStackTrace();
      LOGGER.e("Exception initializing classifier!", e);
      Toast toast =
          Toast.makeText(
              getApplicationContext(), "Classifier could not be initialized", Toast.LENGTH_SHORT);
      toast.show();
      finish();
    }

    previewWidth = size.getWidth();
    previewHeight = size.getHeight();

    sensorOrientation = rotation - getScreenOrientation();
    LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

    LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
    rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
    croppedBitmap = Bitmap.createBitmap(cropSize, cropSize, Config.ARGB_8888);

    frameToCropTransform =
        ImageUtils.getTransformationMatrix(
            previewWidth, previewHeight,
            cropSize, cropSize,
            sensorOrientation, MAINTAIN_ASPECT);

    cropToFrameTransform = new Matrix();
    frameToCropTransform.invert(cropToFrameTransform);

    trackingOverlay = (OverlayView) findViewById(R.id.tracking_overlay);
    trackingOverlay.addCallback(
        new DrawCallback() {
          @Override
          public void drawCallback(final Canvas canvas) {
            tracker.draw(canvas);
            if (isDebug()) {
              tracker.drawDebug(canvas);
            }
          }
        });
  }

  /** Callback for Camera2 API */
  @Override
  public void onImageAvailable(final ImageReader reader) {
    // We need wait until we have some size from onPreviewSizeChosen

    Log.i(TAG,"new Image is Available");
    if (previewWidth == 0 || previewHeight == 0) {
      return;
    }
    if (rgbBytes == null) {
      rgbBytes = new int[previewWidth * previewHeight];
    }
    try {
      final Image image = reader.acquireLatestImage();
      Log.i(TAG,"acquireLatestImage");
      if (image == null) {
        return;
      }
      /*
      if (offloading){
        image.close();
        Log.i(TAG,"during offloading");
        return;
      }
      offloading = true;
      */
      if (isProcessingFrame) {
        Log.i(TAG,"in ProcessingFrame..........");
        image.close();
        return;
      }
      isProcessingFrame = true;

      Trace.beginSection("imageAvailable");
      final Image.Plane[] planes = image.getPlanes();
      fillBytes(planes, yuvBytes);

      tracker.onFrame(
              previewWidth,
              previewHeight,
              planes[0].getRowStride(),
              sensorOrientation,
              yuvBytes[0],
              timestamp);
      trackingOverlay.postInvalidate();

      byte[] originalLuminance = yuvBytes[0];

      yRowStride = planes[0].getRowStride();
      final int uvRowStride = planes[1].getRowStride();
      final int uvPixelStride = planes[1].getPixelStride();
      imageConverter =
              new Runnable() {
                @Override
                public void run() {
                  Log.i(TAG,"converting to RGB");
                  final long startConvertTime = SystemClock.uptimeMillis();
                  ImageUtils.convertYUV420ToARGB8888(
                          yuvBytes[0],
                          yuvBytes[1],
                          yuvBytes[2],
                          previewWidth,
                          previewHeight,
                          yRowStride,
                          uvRowStride,
                          uvPixelStride,
                          rgbBytes);
                  final long latencyconverter = SystemClock.uptimeMillis() - startConvertTime;
                  //Log.i(TAG," Converter time: " + latencyconverter);
                  appendLog(Long.toString(latencyconverter));
                }
              };

      postInferenceCallback =
              new Runnable() {
                @Override
                public void run() {
                  image.close();
                  isProcessingFrame = false;
                  Log.i(TAG,"ProcessingFrame -> false");
                }
              };
      /*------------------ change 2 -------------------*/
      //processImage();
      rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);
      /* luminance */
      if (luminanceCopy == null) {
        luminanceCopy = new byte[originalLuminance.length];
      }
      System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);

      final Canvas canvas = new Canvas(croppedBitmap);
      canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

       //Saving the first frame for checking
      /*
      if (SAVE_PREVIEW) {
        ImageUtils.saveBitmap(rgbFrameBitmap);
        Log.i(TAG,"saving image");
        SAVE_PREVIEW = false;
      }*/
      //readyForNextImage();
      //offloading = false;

      /* Offload croppedBitmap to the server through the socket*/
      runInBackground(
              new Runnable() {
                @Override
                public void run() {
                  frameID += 1;

                  final long StartTime = SystemClock.elapsedRealtime();
                  //appendLog0(Long.toString(StartTime));

                  feedback = offloadImage(croppedBitmap,frameID);
                  if (feedback) {
                    final long allLatency = SystemClock.elapsedRealtime() - StartTime;
                    Log.i(TAG,frameID + " Total latency: " + allLatency);

                    //appendLog(Long.toString(allLatency));

                    feedback = false;
                    Log.i(TAG,"feedback -> true");
                    readyForNextImage();
                  }
                }
              }
      );

      /*--------------------end 2---------------------------*/
    } catch (final Exception e) {
      LOGGER.e(e, "Exception!");
      Trace.endSection();
      return;
    }
    Trace.endSection();
  }


/*
  @Override
  protected void processImage() {
    ++timestamp;
    final long currTimestamp = timestamp;
    byte[] originalLuminance = yuvBytes[0];
    tracker.onFrame(
        previewWidth,
        previewHeight, yRowStride,
        sensorOrientation,
        originalLuminance,
        timestamp);
    trackingOverlay.postInvalidate();

    // No mutex needed as this method is not reentrant.
    if (computingDetection) {
      readyForNextImage();
      return;
    }
    computingDetection = true;
    LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

    rgbFrameBitmap.setPixels(rgbBytes, 0, previewWidth, 0, 0, previewWidth, previewHeight);

    if (luminanceCopy == null) {
      luminanceCopy = new byte[originalLuminance.length];
    }
    System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
    readyForNextImage();

    final Canvas canvas = new Canvas(croppedBitmap);
    canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);

    // For examining the actual TF input.
    if (SAVE_PREVIEW_BITMAP) {
      ImageUtils.saveBitmap(croppedBitmap);
    }

    runInBackground(
        new Runnable() {
          @Override
          public void run() {
            LOGGER.i("Running detection on image " + currTimestamp);
            final long startTime = SystemClock.uptimeMillis();
            final List<Classifier.Recognition> results = detector.recognizeImage(croppedBitmap);
            lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;

            cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
            final Canvas canvas = new Canvas(cropCopyBitmap);
            final Paint paint = new Paint();
            paint.setColor(Color.RED);
            paint.setStyle(Style.STROKE);
            paint.setStrokeWidth(2.0f);

            float minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
            switch (MODE) {
              case TF_OD_API:
                minimumConfidence = MINIMUM_CONFIDENCE_TF_OD_API;
                break;
            }

            final List<Classifier.Recognition> mappedRecognitions =
                new LinkedList<Classifier.Recognition>();

            for (final Classifier.Recognition result : results) {
              final RectF location = result.getLocation();
              if (location != null && result.getConfidence() >= minimumConfidence) {
                canvas.drawRect(location, paint);

                cropToFrameTransform.mapRect(location);

                result.setLocation(location);
                mappedRecognitions.add(result);
              }
            }

            tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);
            trackingOverlay.postInvalidate();

            computingDetection = false;

            runOnUiThread(
                new Runnable() {
                  @Override
                  public void run() {
                    showFrameInfo(previewWidth + "x" + previewHeight);
                    showCropInfo(cropCopyBitmap.getWidth() + "x" + cropCopyBitmap.getHeight());
                    showInference(lastProcessingTimeMs + "ms");
                  }
                });
          }
        });
  }*/

  @Override
  protected int getLayoutId() {
    return R.layout.camera_connection_fragment_tracking;
  }

  @Override
  protected Size getDesiredPreviewFrameSize() {
    return DESIRED_PREVIEW_SIZE;
  }

  // Which detection model to use: by default uses Tensorflow Object Detection API frozen
  // checkpoints.
  private enum DetectorMode {
    TF_OD_API;
  }

  @Override
  protected void setUseNNAPI(final boolean isChecked) {
    runInBackground(() -> detector.setUseNNAPI(isChecked));
  }

  @Override
  protected void setNumThreads(final int numThreads) {
    runInBackground(() -> detector.setNumThreads(numThreads));
  }

  protected void readyForNextImage() {
    if (postInferenceCallback != null) {
      postInferenceCallback.run();
    }
  }

  protected int[] getRgbBytes() {
    imageConverter.run();
    return rgbBytes;
  }

  /** Save Log to file **/
  public void appendLog(String text)
  {
    //String logFile = Environment.getExternalStorageDirectory() + "/TestResults/latency.txt";
    //File logFile = new File("sdcard/TestResults/latency.txt");
    //File logFile = new File(Environment.getExternalStorageDirectory() + "/TestResults/Jou/User/1036/4/latency608.txt");
    File logFile = new File(Environment.getExternalStorageDirectory() + "/INFOCOM2020/convertlatency/latency2649.txt");
    if (!logFile.exists())
    {
      //Log.i(TAA," no this file");
      try
      {
        //Log.i(TAA," create file ");
        logFile.createNewFile();
      }
      catch (IOException e)
      {
        //TODO Auto-generated catch block
        e.printStackTrace();
      }
    }
    try
    {
      //BufferedWriter for performance, true to set append to file flag
      //Log.i(TAA,"write........");
      BufferedWriter buf = new BufferedWriter(new FileWriter(logFile, true));
      buf.append(text);
      buf.newLine();
      buf.flush();
      buf.close();
    }
    catch (IOException e)
    {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
  }

  public void appendLog0(String text)
  {
    //String logFile = Environment.getExternalStorageDirectory() + "/TestResults/starttime.txt";
    //File logFile = new File("sdcard/TestResults/starttime.txt");
    File logFile = new File(Environment.getExternalStorageDirectory() + "/TestResults/Jou/User/1036/4/starttime608.txt");
    if (!logFile.exists())
    {
      //Log.i(TAA," no this file");
      try
      {
        //Log.i(TAA," create file ");
        logFile.createNewFile();
      }
      catch (IOException e)
      {
        //TODO Auto-generated catch block
        e.printStackTrace();
      }
    }
    try
    {
      //BufferedWriter for performance, true to set append to file flag
      //Log.i(TAA,"write........");
      BufferedWriter buf = new BufferedWriter(new FileWriter(logFile, true));
      buf.append(text);
      buf.newLine();
      buf.flush();
      buf.close();
    }
    catch (IOException e)
    {
      // TODO Auto-generated catch block
      e.printStackTrace();
    }
  }

}
