using System;
using System.IO;
using System.Threading.Tasks;
using Microsoft.ML;
using Microsoft.ML.Data;
using Microsoft.ML.Models;
using Microsoft.ML.Trainers;
using Microsoft.ML.Transforms;

namespace AbaloneAgeApp
{
    class Program
    {
        // REF: Machine Learning using ML.NET and its integration into ASP.NET Core Web application
        // https://blogs.msdn.microsoft.com/uk_faculty_connection/2018/06/22/machine-learning-using-ml-net-and-its-integration-into-asp-net-core-web-application/
        static async Task Main(string[] args)
        {
            Console.WriteLine("Hello World!");
            var model = await Train();

            var abaloneSample = new Abalone()
            {
                Length = 0.524f,        //mm
                Diameter = 0.408f,      //mm
                Height = 0.140f,        //mm
                WholeWeight = 0.829f,   //grames
                ShellWeight = 0.239f,   //grames
                Age = 0                 //must be 9
            };

            var prediction = model.Predict(abaloneSample);
            Console.WriteLine("Predicted Age is {0}", Math.Floor(prediction.Age));
        }

        public static async Task<PredictionModel<Abalone, AbaloneAgePrediction>> Train()
        {
            // 預測資料來源存放位址
            var datapath = $"{Directory.GetCurrentDirectory()}/Data/abalone.data";
            // 預測資料結果存放位址
            var modelpath = $"{Directory.GetCurrentDirectory()}/Data/Model.zip";

            // STEP 2: 建立執行預測運算的 Pipe Line
            var pipeline = new LearningPipeline{
                // 從預測資料來源中讀取資料，每一行為一筆資料，設定使用 ',' 當作分隔字元
                new TextLoader(datapath).CreateFrom<Abalone>(separator: ','),
                // 訓練預測模型時，將 Age 當作
                // when model is trained, values under Column Label are considered as correct values and as we want to predict Age we should copy Age column into Label Column.
                new ColumnCopier(("Age", "Label")),
                // Values in Column Sex are M or F, however algorithm requires numeric values, therefore this function assigns them different numeric values and makes suitable for training model.s
                new CategoricalOneHotVectorizer("Sex"),
                // this is the function in which we tell pipeline which features to include to predict Age of the Abalone. We must decide how relevant are the features for our calculation and based on that decide whether include it or not. Only those features will be included in learning process, whose names are declared in this function. As you can see I have excluded Shucked Weight, and Viscera Weight, even though they are absolutely relevant and will probably make calculation a bit more accurate, they are not easy to obtain. Our aim is to make predictions based on easily obtainable measures, so we can quickly check how old tha Abalone is. To make things even easier, we can exclude Sex feature, as it is not very intuitive to tell whether Abalone is male or female, and it doesn’t have massive influence on our prediction.
                new ColumnConcatenator(
                    "Features",
                    "Sex",
                    "Length",
                    "Diameter",
                    "Height",
                    "WholeWeight",
                    "ShellWeight"),
                // finally we define the algorithm that we want to use in the learning process, I will not go in too much detail here but if you want to find out more about Tree Regression algorithms, you can read this article.
                new FastTreeRegressor()
            };

            var model = pipeline.Train<Abalone, AbaloneAgePrediction>();
            await model.WriteAsync(modelpath);
            return model;
        }
    }
}
