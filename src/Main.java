import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;
import java.util.stream.IntStream;

public class Main {
    static double TRAINING_DATA[][][] = new double[][][]{
            {{0, 0, 0, 0},{0}},
            {{0, 0, 0, 1},{1}},
            {{0, 0, 1, 0},{1}},
            {{0, 0, 1, 1},{0}},
            {{0, 1, 0, 0},{1}},
            {{0, 1, 0, 1},{0}},
            {{0, 1, 1, 0},{0}},
            {{0, 1, 1, 1},{1}},
            {{1, 0, 0, 0},{1}},
            {{1, 0, 0, 1},{0}},
            {{1, 0, 1, 0},{0}},
            {{1, 0, 1, 1},{1}},
            {{1, 1, 0, 0},{0}},
            {{1, 1, 0, 1},{1}},
            {{1, 1, 1, 0},{1}},
            {{1, 1, 1, 1},{0}}};
    public static void main (String[] args){



            for(int i=0; i< 10; i++) {

                run();
                NeuralNetwork.LEARNING_RATE+=0.05;
            }
    //System.out.println(0.9*0.0);

    }
public static long run(){
        NeuralNetwork NN= new NeuralNetwork();
        // run the forwarding computation
        double[] result = new double[16];
        Arrays.fill(result, 0);
//        IntStream.range(0,Main.TRAINING_DATA.length).forEach(i ->{
//            result[i] = NN.forwardComputation(Main.TRAINING_DATA[i][0])
//                    .getNeurons()[NeuralNetwork.INPUT_NEURONS+NeuralNetwork.HIDDEN_NEURONS]
//                    .getOutput();
//        });
//        printResult(result);
        boolean flag = true;
        long epochs =0;
        while(flag){
           // System.out.println("[epoch "+ epochs+ " ]");
            IntStream.range(0, Main.TRAINING_DATA.length).forEach(i->{
//                result[i] = NN.forwardComputation(Main.TRAINING_DATA[i][0])
//                        .backwardComputation(Main.TRAINING_DATA[i][1][0])
//                        .getNeurons()[NeuralNetwork.INPUT_NEURONS+NeuralNetwork.HIDDEN_NEURONS]
//                        .getOutput();
               System.out.println( NN.forwardComputation(Main.TRAINING_DATA[i][0])
                        .backwardComputation(Main.TRAINING_DATA[i][1][0]));
                result[i] = NN.forwardComputation(Main.TRAINING_DATA[i][0])
                        .getNeurons()[NeuralNetwork.INPUT_NEURONS+NeuralNetwork.HIDDEN_NEURONS]
                        .getOutput();


            });
            int count =0;
            for(int i=0; i < result.length; i++){
                if(Math.abs(result[i]- Main.TRAINING_DATA[i][1][0]) <=0.05){
                    count++;
                }
            }
            if(count == 15){
                flag = false;
            }
            epochs++;
        }

        System.out.println("With LEARNING RATE = "+ String.format("%.2f",NeuralNetwork.LEARNING_RATE)+ " number of epochs is "+ epochs);
        return epochs;
    }
    static void printResult(double[]result){
        System.out.println("     Input1    |    Input2    |     Input3    |     Input3     |    Target Result   |  Result      ");
        System.out.println("----------------------------------------------------------------------------------------------");
        IntStream.range(0, TRAINING_DATA.length).forEach(i ->{
            IntStream.range(0,TRAINING_DATA[0][0].length).forEach(j-> System.out.print("      "+TRAINING_DATA[i][0][j]+"      |"));
            System.out.println("      "+TRAINING_DATA[i][1][0]+"      |      " +String.format("%.5f",result[i])+"      ");
        });
    }

}
