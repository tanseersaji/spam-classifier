// 1 = not Spam
// 0 = Spam

import java.io.IOException;
import java.math.BigDecimal;
import java.math.RoundingMode;
import java.time.Instant;

public class LearningModel {

    private int trainingSetLimit = (int) (4326 * 0.6);
    private int CVSetLimit = trainingSetLimit + (int) (4326 * 0.2);
    private double[] Theta = matrix.random(1943);
    private double alpha = 4;


    dataVectorization dv = new dataVectorization();

    private double[][] getXTrainingSet() throws IOException {
        double[][] XTrain = new double[trainingSetLimit][1943];
        for(int i = 0; i < trainingSetLimit;i++){
            String file = "E_Train/TRAIN_";
            file = file.concat(String.format("%05d", i));
            file = file.concat(".eml");
            int temp[];
            temp =  dv.getDataVector(file);
            for(int j = 0;j < temp.length; j++){
                XTrain[i][j] = temp[j];
            }

        }
        return XTrain;
    }
    private double[][] getXCrossValidationSet() throws IOException {
        double[][] Xcv = new double[(int) (4326 * 0.2)][1943];
        for(int i = trainingSetLimit; i < CVSetLimit;i++){
            String file = "E_Train/TRAIN_";
            file = file.concat(String.format("%05d", i));
            file = file.concat(".eml");
            int temp[] =  dv.getDataVector(file);
            for(int j = 0;j < temp.length; j++){
                Xcv[i - trainingSetLimit][j] = temp[j];
            }

        }
        return Xcv;
    }
    private double[][] getXTestingSet() throws IOException {

        double[][] Xtest = new double[(int) (4326 * 0.2)+2][1943];
        for(int i = CVSetLimit; i <= 4326;i++){
            String file = "E_Train/TRAIN_";
            file = file.concat(String.format("%05d", i));
            file = file.concat(".eml");
            int temp[] =  dv.getDataVector(file);
            for(int j = 0;j < temp.length; j++){
                Xtest[i - CVSetLimit][j] = temp[j];
            }

        }
        return Xtest;
    }

    private double[]  getYtrain() throws IOException {

        int[] fullLabel = dv.getLabelArray();
        double[] Ytrain = new double[(int) (4326 * 0.6)];
        for(int i=0; i < trainingSetLimit;i++){
            Ytrain[i] = fullLabel[i];
        }

        return Ytrain;
    }

    private double[] getYcv() throws IOException {
        int[] fullLabel = dv.getLabelArray();
        double[] Ycv = new double[(int) (4326 * 0.2)];
        for(int i=trainingSetLimit; i < CVSetLimit;i++){
            Ycv[i - trainingSetLimit] = fullLabel[i];
        }

        return Ycv;
    }

    private double[] getYtest() throws IOException {
        int[] fullLabel = dv.getLabelArray();
        double[] Ytest = new double[(int) (4326 * 0.2)+2];
        for(int i=CVSetLimit; i <= 4326;i++){
            Ytest[i - CVSetLimit] = fullLabel[i];
        }
        return Ytest;
    }

    private static double round(double value, int places) {
        if (places < 0) throw new IllegalArgumentException();
        BigDecimal bd = new BigDecimal(value);
        bd = bd.setScale(places, RoundingMode.HALF_UP);
        return bd.doubleValue();
    }

    // returns Cross of matrix X and Vector Theta
    private double[]  hypothesis(double[][] X,double[] theta){
        return matrix.multiply(X,theta);
    }
    private double  hypothesis(double[] X,double[] theta){
        return matrix.multiply(X,theta);
    }

    // returns Cost of Hypothesis ( sigmoid(X*Theta)
    private double cost(double[] h, double[] y){
        return (1.0/h.length) * matrix.subtract( matrix.multiply(matrix.multiply(-1,y),matrix.log
                (h)),matrix.multiply(matrix.subtract(matrix.ones(y.length),y),matrix.log(matrix.subtract(matrix.ones
                (h.length),h))) );
    }

    // returns Partial Derivation of Cost Funtion
    private double[] costPrime(double[][] X,double[] h, double[] y ){
        return matrix.multiply((1.0/h.length), matrix.multiply(matrix.transpose(X),matrix.subtract(h,y)));
    }

    // Performs Gradient Decent to optimize Theta (weights)
    private void gradientDecent(double[][] X, double[] y){
        double[] h = matrix.sigmoid(matrix.multiply(X,Theta));
        double[] cp = costPrime(X,h,y);

        double scp = 0;
        for(double i: cp){
            scp = scp + i;
        }
        System.out.println("Initial Cost : " + cost(h,y));
        // Average Cost
        scp = round(scp/h.length,5);
        System.out.println("Initial Cost = " + scp);
        int j = 0;

        // while Average Cost is not 0
       while(scp > 0.00001){
            Theta =  matrix.subtract(Theta,matrix.multiply(alpha,cp));
            h = matrix.sigmoid(matrix.multiply(X,Theta));
            cp = costPrime(X,h,y);
            double sum = 0;
            for(double i: cp){
                sum = sum + i;
            }
            scp = round(sum/h.length,5);
            System.out.println("Iteration = " + j++ + " Cost = " + scp);
        }
    }


    // Trains to model
    public void train(){
        double[][] Xtrain = new double[(int) (4326 * 0.6)][1943];
        double[] Ytrain = new double[(int) (4326 * 0.6)];


        try {
             Xtrain = getXTrainingSet();
             Ytrain = getYtrain();
        } catch (IOException e) {
            e.printStackTrace();
        }

        long t0 = Instant.now().toEpochMilli();

        gradientDecent(Xtrain,Ytrain);
        System.out.println("Model trained in  "+ (Instant.now().toEpochMilli() - t0) +" millisec");


        crossValidate();
        test();

    }

    private void crossValidate(){
        double[][] Xcv = new double[(int)(4326 * 0.2)][1943];
        double[] Ycv = new double[(int) (4326 * 0.2)];
        try {
             Xcv = getXCrossValidationSet();
                Ycv = getYcv();
        } catch (IOException e) {
            e.printStackTrace();
        }

        double[] h = matrix.sigmoid(hypothesis(Xcv,Theta ));

        double tp = 0,tn = 0,fp=0,fn=0;
        for(int i=0;i<h.length;i++){
            int hi = (h[i] >= 0.9) ? 1 : 0;

            if(hi == 1 && Ycv[i] == 1){
                tp++;
            }
            else if(hi == 1 && Ycv[i] == 0){
                fp++;
            }
            else if(hi == 0 && Ycv[i] == 0){
                tn++;
            }
            else if(hi == 0 && Ycv[i] == 1){
                fn++;
            }
        }

        double P = tp/(tp + fp);
        double R = tp/(tp + fn);
        double F1 = 2*(P*R)/(P+R);
        double accuracy = (tp + tn)/h.length;

        System.out.println("Cross Validation Cost = " + cost(h,Ycv));
        System.out.println("CV Precision = " + P);
        System.out.println("CV Recall = " + R);
        System.out.println("CV  F1 Score = " + F1);
        System.out.println("CV Accuracy = " + accuracy*100);

    }

    private void test(){
        double[][] Xtest = new double[(int)(4326 * 0.2)][1943];
        double[] Ytest = new double[(int) (4326 * 0.2)];
        try {
            Xtest = getXTestingSet();
            Ytest = getYtest();
        } catch (IOException e) {
            e.printStackTrace();
        }

        double[] h = matrix.sigmoid(hypothesis(Xtest,Theta));
        double tp = 0,tn = 0;
        for(int i=0;i<h.length;i++){

            int hi = (h[i] >= 0.9) ? 1 : 0;

            if(hi == 1 && Ytest[i] == 1){
                tp++;
            }
            else if(hi == 0 && Ytest[i] == 0){
                tn++;
            }
        }
        double accuracy = (tp + tn)/h.length;
        System.out.println("Testing Cost : " + cost(h,Ytest));
        System.out.println("Testing Accuracy = " + accuracy*100);
    }

    // Classifies Spam or Not Spsm
    public void predict(String filename){
        double[] dataVector = new double[1943];
        try {
            int[] temp = dv.getDataVector(filename);
            for (int i=0;i<temp.length;i++){
                dataVector[i] = temp[i];
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
        double h = matrix.sigmoid(hypothesis(dataVector,Theta));
        System.out.println((h>=0.9)?"Not Spam":"Spam");
    }
}
