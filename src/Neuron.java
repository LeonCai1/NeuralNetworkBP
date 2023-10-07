
public class Neuron {
    private NeuralNetwork.LayerType layerType;
    private double a = 1;
    // get weights from random number between -1 and 1
    private double[] weights = {Math.random()*(Math.random() > 0.5? 1: -1),
            Math.random()*(Math.random() > 0.5? 1: -1 ),
            Math.random()*(Math.random() > 0.5? 1: -1 ),
            Math.random()*(Math.random() > 0.5? 1: -1 )};
    private double[] changeOfWeights = {0.0,0.0,0.0,0.0};
    private double bias = Math.random()*(Math.random() > 0.5? 1: -1);
    private double output = 0;
    private double error = 0;
    // constructor for neuron
    public Neuron(NeuralNetwork.LayerType layerType){
        this.layerType = layerType;
    }
    public NeuralNetwork.LayerType getLayerType() {return layerType;}

    public double[] getWeights() {return weights;}
    public double[] getWeightsChange() {return changeOfWeights;}
    public void setError(double error) {this.error = error;}

    public double getError() {return error;}

    public double getOutput() {return output;}

    public double getBias() {
        return bias;
    }

    public void setBias(double bias) {
        this.bias = bias;
    }

    public void setOutput(double output) {this.output = output;}

    public void applyActivationFunction(double weightedSum){
        output = 1/(1+ Math.exp(-1*a*weightedSum));
    }
    public double derivativeOfSig(){
        return a*output*(1-output);
    }

    public String toString() {
        StringBuilder sb = new StringBuilder();
        if(layerType == NeuralNetwork.LayerType.I){
            sb.append("(").append(layerType).append(": ")
                    .append(String.format("%.2f", output)).append(")");
        }
        else{
            sb.append("(").append(layerType).append(": ")
                    .append(String.format("%.2f", weights[0])).append(", ")
                    .append(String.format("%.2f", weights[1])).append(", ")
                    .append(String.format("%.2f", bias)).append(", ")
                    .append(String.format("%.2f", output)).append(")");

        }

        return sb.toString();
    }
}
