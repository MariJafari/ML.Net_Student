using Microsoft.ML;
using Microsoft.ML.Data;

class Program
{
    static void Main(string[] args)
    {
        // Paths for the data and the model
        string _dataPath = Path.Combine(Environment.CurrentDirectory, "Data", "Student.csv");
        string _modelPath = Path.Combine(Environment.CurrentDirectory, "Data", "StudentKnowledgeModel.zip");

        // Initialize MLContext
        var mlContext = new MLContext(seed: 0);

        // Load the data
        IDataView dataView = mlContext.Data.LoadFromTextFile<StudentData>(_dataPath, hasHeader: true, separatorChar: ',');

        // Data process configuration with pipeline data transformations 
        var dataProcessPipeline = mlContext.Transforms.Conversion.MapValueToKey("UNS", "Label")
            .Append(mlContext.Transforms.Concatenate("Features", nameof(StudentData.STG), nameof(StudentData.SCG), nameof(StudentData.STR), nameof(StudentData.LPR), nameof(StudentData.PEG)));

        // Set the training algorithm 
        var trainer = mlContext.MulticlassClassification.Trainers.SdcaMaximumEntropy(labelColumnName: "Label", featureColumnName: "Features")
            .Append(mlContext.Transforms.Conversion.MapKeyToValue("PredictedLabel"));
        var trainingPipeline = dataProcessPipeline.Append(trainer);

        // Train the model fitting to the DataSet
        Console.WriteLine("Training the model...");
        var trainedModel = trainingPipeline.Fit(dataView);

        // Saving the model to a file
        using (var fs = new FileStream(_modelPath, FileMode.Create, FileAccess.Write, FileShare.Write))
        {
            mlContext.Model.Save(trainedModel, dataView.Schema, fs);
            Console.WriteLine("Model saved to " + _modelPath);
        }

        // Create a prediction engine from the model for making single predictions
        var predictionEngine = mlContext.Model.CreatePredictionEngine<StudentData, KnowledgePrediction>(trainedModel);

        // Create sample data to test prediction
        // This should ideally be replaced with real data for actual predictions
        var sampleStudent = new StudentData
        {
            STG = 0.5f, // Sample data
            SCG = 0.5f,
            STR = 0.5f,
            LPR = 0.5f,
            PEG = 0.5f
        };

        // Make a prediction
        var predictionResult = predictionEngine.Predict(sampleStudent);

        // Print the prediction result
        Console.WriteLine($"Predicted Knowledge Level: {predictionResult.PredictedKnowledgeLevel}");
    }
}

public class StudentData
{
    public float STG;
    public float SCG;
    public float STR;
    public float LPR;
    public float PEG;
    public string UNS; // This is the label in our case
}

public class KnowledgePrediction
{
    public string PredictedKnowledgeLevel;
    public float[] Score;
}
