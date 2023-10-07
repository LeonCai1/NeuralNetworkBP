import java.util.Arrays;
import java.util.stream.IntStream;

public class NeuralNetwork {
     static enum LayerType  {I, H, O};
     static double LEARNING_RATE = 0.05;
     final static int INPUT_NEURONS = 4;
     final static int HIDDEN_NEURONS = 4;
     final static int OUTPUT_NEURONS = 1;
     private Neuron[] neurons = new Neuron[INPUT_NEURONS+HIDDEN_NEURONS+OUTPUT_NEURONS];
     public NeuralNetwork(){
          IntStream.range(0, INPUT_NEURONS).forEach(i -> neurons[i] = new Neuron(LayerType.I));
          IntStream.range(INPUT_NEURONS, INPUT_NEURONS+HIDDEN_NEURONS).forEach(i -> neurons[i] = new Neuron(LayerType.H));
          neurons[INPUT_NEURONS+HIDDEN_NEURONS] = new Neuron(LayerType.O);
     }
     public NeuralNetwork forwardComputation(double input[]){
          double weightedSum = 0;
          for(int i=0; i < neurons.length; i++){
               switch (neurons[i].getLayerType()){
                    case I:
                         neurons[i].setOutput(input[i]);
                         break;
                    case H:
                         // v(n) = ∑ wi*xi + b
                         for(int j=0; j < INPUT_NEURONS;j++ ){
                              weightedSum+= neurons[i].getWeights()[j] * neurons[j].getOutput();
                         }
                         weightedSum+=neurons[i].getBias();
                         // calculate φ(v)
                         neurons[i].applyActivationFunction(weightedSum);
                         break;
                    case O:
                         for(int j = INPUT_NEURONS; j < INPUT_NEURONS+HIDDEN_NEURONS; j++){
                              weightedSum+= neurons[i].getWeights()[j-HIDDEN_NEURONS] * neurons[j].getOutput();
                         }
                         weightedSum+=neurons[i].getBias();
                         neurons[i].applyActivationFunction(weightedSum);
                         break;
               }
          }
          return this;
     }
     public NeuralNetwork backwardComputation(double desiredResult){
          // ******For the output layer:*******
          // local gradient = error signal* derivative of output signal
          //                  where error signal = (desired output - actual actual output)
          neurons[neurons.length-1].setError((desiredResult-neurons[neurons.length-1].getOutput())
                                                           * neurons[neurons.length-1].derivativeOfSig());
          // update bias
          neurons[neurons.length-1].setBias(neurons[neurons.length-1].getBias()+ LEARNING_RATE*neurons[neurons.length-1].getError());
          // update weights
          //  w_j(n+1) = w_j(n) +Learning_RATE * error signal(n) * weightedSum(n)
          for(int i=0; i < HIDDEN_NEURONS; i++){
               double PrevWeight = neurons[neurons.length-1].getWeights()[i];
               neurons[neurons.length-1].getWeights()[i]+= (LEARNING_RATE* neurons[neurons.length-1].getError()
                                                                      *neurons[INPUT_NEURONS+i].getOutput()+0.9 * neurons[neurons.length-1].getWeightsChange()[i]);
               neurons[neurons.length-1].getWeightsChange()[i] = neurons[neurons.length-1].getWeights()[i]-PrevWeight;
          }
          //******For the hidden layer: *******
          //
          for(int i=INPUT_NEURONS+HIDDEN_NEURONS-1; i >= INPUT_NEURONS; i--){
               // error_signal = derivative of sigmoid_function * error_signal of l+1 layer(output layer) * weight of output layer
               neurons[i].setError(neurons[i].derivativeOfSig()*neurons[neurons.length-1].getError()
                       *neurons[neurons.length-1].getWeights()[i-HIDDEN_NEURONS]);
               // update bias
               neurons[i].setBias(neurons[i].getBias()+LEARNING_RATE*neurons[i].getError());
               // update weights
               // w_ji(n+1) = w_ji(n) + alpha[delta(w_ji(n-1)) + LEARNING_RATE * error_signal(n) * weighted_sum_0f_previous_layer(n)
               for(int j=0; j < INPUT_NEURONS; j++){
                    //TODO: adding momentumn
                    double PrevWeight = neurons[i].getWeights()[j];
                   // neurons[i].getWeightsChange()[j] = (LEARNING_RATE * neurons[i].getError()*neurons[j].getOutput()+ 0.9 *PrevChangeOfWeight);
                    neurons[i].getWeights()[j]+=(LEARNING_RATE * neurons[i].getError()*neurons[j].getOutput()+ 0.9 * neurons[i].getWeightsChange()[j]);
                    neurons[i].getWeightsChange()[j] = neurons[i].getWeights()[j]-PrevWeight;

               }
          }


          return this;
     }
     public Neuron[] getNeurons() {
          return neurons;
     }

     public String toString() {
          return Arrays.toString(neurons);
     }
}
