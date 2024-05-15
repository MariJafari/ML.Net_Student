using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Microsoft.ML.Data;

namespace Question3
{

    public class Data
    {
        [LoadColumn(0)]
        public float STG; // The degree of study time for goal object materials

        [LoadColumn(1)]
        public float SCG; // The degree of repetition number of user for goal object materials

        [LoadColumn(2)]
        public float STR; // The degree of study time of user for related objects with goal object

        [LoadColumn(3)]
        public float LPR; // The exam performance of user for related objects with goal object

        [LoadColumn(4)]
        public float PEG; // The exam performance of user for goal objects

        // Assuming the UNS field is the one you're trying to predict,
        // and it's the last column in your CSV
        [LoadColumn(5)]
        public string UNS; // The knowledge level of user
    }

    public class KnowledgePrediction
    {
        [ColumnName("PredictedLabel")]
        public string PredictedKnowledgeLevel;

        [ColumnName("Score")]
        public float[] Score;
    }
}