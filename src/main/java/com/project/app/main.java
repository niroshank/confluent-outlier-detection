package com.project.app;

import hex.genmodel.easy.RowData;
import hex.genmodel.easy.EasyPredictModelWrapper;
import hex.genmodel.easy.prediction.*;

import java.util.Arrays;

public class main {
  private static String modelClassName = "io.confluent.ksql.function.udf.ml"
          + ".DeepLearning_model_python_1639244444917_7";

  public static void main(String[] args) throws Exception {
    hex.genmodel.GenModel rawModel;
    rawModel = (hex.genmodel.GenModel) Class.forName(modelClassName).newInstance();
    EasyPredictModelWrapper model = new EasyPredictModelWrapper(rawModel);

    String sensorinput = "46.18055# 52.30035# 44.53125# 631.4814# 73.71378# 13.252310000000001# 16.16753# 15.94329# 15.08247# 39.40287# 53.06762# 36.183479999999996# 1.5380719999999999# 421.3417# 463.3826# 462.7775# 2.5470669999999997# 653.3826# 390.921# 859.4838# 449.281# 950.8918# 610.1232# 698.5195# 850.2299# 580.2689# 737.357# 698.5954# 634.7222# 719.2708# 723.1457# 441.0215# 172.8622# 332.9419# 199.9485# 101.2788# 44.53125# 32.29166# 67.44791# 31.77083206176761# 32.29166# 38.80208# 56.13426# 50.34722# 46.875# 38.77315# 190.9722# 112.5579#";
    String[] inputStringArray = sensorinput.split("#");
    double[] doubleValues = Arrays.stream(inputStringArray)
            .mapToDouble(Double::parseDouble)
            .toArray();

//    java.util.Random rng = new java.util.Random();
    RowData row = new RowData();
    int j = 0;
    for (String colName : rawModel.getNames()) {
      row.put(colName,doubleValues[j]);
      j++;
    }

    AutoEncoderModelPrediction p = model.predictAutoEncoder(row);
    System.out.println("detection: " + row);
    System.out.println("mse: " + p.mse);
    System.out.println("original: " + java.util.Arrays.toString(p.original));
    System.out.println("reconstructedrowData: " + p.reconstructedRowData);
    System.out.println("reconstructed: " + java.util.Arrays.toString(p.reconstructed));

    double sum = 0;
    for (int i = 0; i<p.original.length; i++) {
      sum += (p.original[i] - p.reconstructed[i])*(p.original[i] - p.reconstructed[i]);
    }
    double mse = sum/p.original.length;
    System.out.println("MSE: " + mse);
  }
}