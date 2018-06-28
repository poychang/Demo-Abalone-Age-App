using Microsoft.ML.Runtime.Api;

namespace AbaloneAgeApp
{
    // STEP 1: 定義資料模型

    /// <summary>
    /// 定義資料模型。Abalone 模型是做為訓練資料的資料模型，
    /// 除了最後一項 Age，其他屬性將可以預測欄位名稱為 Label 的值。
    /// </summary>
    public class Abalone
    {
        // 裝飾子
        [Column("0")]
        public string Sex;
        [Column("1")]
        public float Length;
        [Column("2")]
        public float Diameter;
        [Column("3")]
        public float Height;
        [Column("4")]
        public float WholeWeight;
        [Column("5")]
        public float SuckedWeight;
        [Column("6")]
        public float VisceraWeight;
        [Column("7")]
        public float ShellWeight;
        [Column("8")]
        public float Age;
    }

    /// <summary>
    /// 定義預測結果的資料模型。預測運算後的結果會用 AbaloneAgePrediction 模型來表達。
    /// </summary>
    public class AbaloneAgePrediction
    {
        [ColumnName("Score")]
        public float Age;
    }
}
