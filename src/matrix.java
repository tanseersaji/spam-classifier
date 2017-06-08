
public class matrix {

    // return a random m-by-n matrix with values between 0 and 1
    public static double[][] random(int m, int n) {
        double[][] a = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                a[i][j] = Math.random();
        return a;
    }
    public static double[] random(int m){
        double[] a = new double[m];
        for (int i = 0;i<m;i++){
            a[i] = Math.random();
        }
        return a;
    }

    // return n-by-n identity matrix I
    public static double[][] identity(int n) {
        double[][] a = new double[n][n];
        for (int i = 0; i < n; i++)
            a[i][i] = 1;
        return a;
    }

    // return vector of ones
    public static double[] ones(int n){
        double[] ones = new double[n];
        for (int i = 0;i<n;i++){
            ones[i] = 1;
        }
        return ones;
    }

    // return x^T y
    public static double multiply(double[] x, double[] y) {
        if (x.length != y.length) throw new RuntimeException("Illegal vector dimensions.");
        double sum = 0.0;
        for (int i = 0; i < x.length; i++)
            sum += x[i] * y[i];
        return sum;
    }

    // return B = A^T
    public static double[][] transpose(double[][] a) {
        int m = a.length;
        int n = a[0].length;
        double[][] b = new double[n][m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                b[j][i] = a[i][j];
        return b;
    }

    // return c = a + b
    public static double[][] add(double[][] a, double[][] b) {
        int m = a.length;
        int n = a[0].length;
        double[][] c = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                c[i][j] = a[i][j] + b[i][j];
        return c;
    }

    // return c = a + b
    public static double[] add(double[] a,double[] b){
        int m = a.length;
        double[] c = new double[m];

        for (int i=0;i<m;i++){
            c[i] = a[i] + b[i];
        }
        return c;
    }

    // return c = a - b
    public static double[][] subtract(double[][] a, double[][] b) {
        int m = a.length;
        int n = a[0].length;
        double[][] c = new double[m][n];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                c[i][j] = a[i][j] - b[i][j];
        return c;
    }

    // returns c = a - b
    public static double[] subtract(double[] a,double[] b){
        int m = a.length;
        double[] c = new double[m];

        for (int i=0;i<m;i++){
            c[i] = a[i] - b[i];
        }
        return c;
    }

    // return c = a * b
    public static double[][] multiply(double[][] a, double[][] b) {
        int m1 = a.length;
        int n1 = a[0].length;
        int m2 = b.length;
        int n2 = b[0].length;
        if (n1 != m2) throw new RuntimeException("Illegal matrix dimensions.");
        double[][] c = new double[m1][n2];
        for (int i = 0; i < m1; i++)
            for (int j = 0; j < n2; j++)
                for (int k = 0; k < n1; k++)
                    c[i][j] += a[i][k] * b[k][j];
        return c;
    }

    // matrix-vector multiplication (y = A * x)
    public static double[] multiply(double[][] a, double[] x) {
        int m = a.length;
        int n = a[0].length;
        if (x.length != n) throw new RuntimeException("Illegal matrix dimensions.");
        double[] y = new double[m];
        for (int i = 0; i < m; i++)
            for (int j = 0; j < n; j++)
                y[i] += a[i][j] * x[j];
        return y;
    }


    // vector-matrix multiplication (y = x^T A)
    public static double[] multiply(double[] x, double[][] a) {
        int m = a.length;
        int n = a[0].length;
        if (x.length != m) throw new RuntimeException("Illegal matrix dimensions.");
        double[] y = new double[n];
        for (int j = 0; j < n; j++)
            for (int i = 0; i < m; i++)
                y[j] += a[i][j] * x[i];
        return y;
    }

    // scalar-vector multiplication
    public static double[] multiply(double s,double[] x){
        double[] c = new double[x.length];
        for(int i = 0;i<x.length;i++){
            c[i] = s*x[i];
        }
        return c;
    }

    // scalar-matrix multiplication
    public static double[][] multiply(double s,double[][] X){
        double[][] c = new double[X.length][X[0].length];
        for (int i = 0;i<X.length;i++){
            for (int j = 0;j<X[0].length;j++){
                c[i][j] = s*X[i][j];
            }
        }
        return c;
    }

    // return natural log of each element in a vector.
    public static double[] log(double[] h){
        double[] l = new double[h.length];
        for (int i = 0;i<h.length;i++){
            l[i] = Math.log1p(h[i]);
        }
        return l;
    }


    // return sigmoid function of each element in a vector.
    public static double[] sigmoid(double[] h){
        double[] s = new double[h.length];
        for (int i = 0;i<h.length;i++){
            s[i] = 1/(1 + Math.exp(-h[i]));
        }
        return s;
    }

    // I made a terrible mistake and I'm too lazy to fix that. So thats why I'm making this Method :P
    public static double subtract(double a,double b){
        return a - b;
    }
    public static double sigmoid(double h){
        return 1/(1 + Math.exp(-h));
    }



}
