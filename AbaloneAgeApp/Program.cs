using Microsoft.ML.Legacy;
using Microsoft.ML.Legacy.Data;
using Microsoft.ML.Legacy.Trainers;
using Microsoft.ML.Legacy.Transforms;
using System;
using System.IO;
using System.Threading.Tasks;

namespace AbaloneAgeApp
{
    public class Program
    {
        // REF: Machine Learning using ML.NET and its integration into ASP.NET Core Web application
        // https://blogs.msdn.microsoft.com/uk_faculty_connection/2018/06/22/machine-learning-using-ml-net-and-its-integration-into-asp-net-core-web-application/
        private static async Task Main()
        {
            var model = await Train();
            var abaloneSample = new Abalone()
            {
                Length = 0.524f,        // mm
                Diameter = 0.408f,      // mm
                Height = 0.140f,        // mm
                WholeWeight = 0.829f,   // grames
                ShellWeight = 0.239f,   // grames
                Age = 0                 // must be 9
            };
            var prediction = model.Predict(abaloneSample);

            Console.WriteLine("Predicted Age is {0}", Math.Floor(prediction.Age));
        }

        public static async Task<PredictionModel<Abalone, AbaloneAgePrediction>> Train()
        {
            // 預測資料來源存放位置
            var dataPath = $"{Directory.GetCurrentDirectory()}/Data/abalone.data.txt";
            // 預測資料結果存放位置
            var modelPath = $"{Directory.GetCurrentDirectory()}/Data/Model.zip";

            // STEP 2: 建立執行預測運算的 Pipeline
            var pipeline = new LearningPipeline{
                // 從數據集中讀取資料，每一行為一筆資料，使用 ',' 當作分隔字元
                new TextLoader(dataPath).CreateFrom<Abalone>(separator: ','),
                // 訓練預測模型時，將 Age 當作
                // when model is trained, values under Column Label are considered as correct values and as we want to predict Age we should copy Age column into Label Column.
                new ColumnCopier(("Age", "Label")),
                // Values in Column Sex are M or F, however algorithm requires numeric values, therefore this function assigns them different numeric values and makes suitable for training model.s
                new CategoricalOneHotVectorizer("Sex"),
                // this is the function in which we tell pipeline which features to include to predict Age of the Abalone. We must decide how relevant are the features for our calculation and based on that decide whether include it or not. Only those features will be included in learning process, whose names are declared in this function. As you can see I have excluded Shucked Weight, and Viscera Weight, even though they are absolutely relevant and will probably make calculation a bit more accurate, they are not easy to obtain. Our aim is to make predictions based on easily obtainable measures, so we can quickly check how old tha Abalone is. To make things even easier, we can exclude Sex feature, as it is not very intuitive to tell whether Abalone is male or female, and it does not have massive influence on our prediction.
                new ColumnConcatenator(
                    "Features",
                    "Sex",
                    "Length",
                    "Diameter",
                    "Height",
                    "WholeWeight",
                    "ShellWeight"),
                // 最後我們設定要使用的學習器演算法，這裡使用 Tree Regression 決策樹演算法
                new FastTreeRegressor()
            };

            var model = pipeline.Train<Abalone, AbaloneAgePrediction>();
            await model.WriteAsync(modelPath);

            return model;
        }
    }
}
