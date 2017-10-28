using cl.uv.leikelen.Module.Processing.EEGEmotion2Channels;
using cl.uv.leikelen.Module.Processing.EEGEmotion2Channels.View;
using System;
using System.Collections.Generic;
using System.IO;

namespace projectMH
{
    public class MainClass
    {
        public static void Main(string[] args)
        {
            int numberExecutions = EEGEmoProc2ChSettings.Instance.Executions.Value;
            string assemblyPath = System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetEntryAssembly().Location);
            if (!Directory.Exists(Path.Combine(assemblyPath, "salida")))
            {
                Directory.CreateDirectory(Path.Combine(assemblyPath, "salida"));
            }
            string dataPath = Path.Combine(assemblyPath, "data");
            var dict = new Dictionary<TagType, List<List<double[]>>>();
            Console.WriteLine("Leyendo HALV File 1");
            dict.Add(TagType.HALV, TrainerFileSelector.ReadSqlite(Path.Combine(dataPath, EEGEmoProc2ChSettings.Instance.HALVFileName.Value)));
            Console.WriteLine("Leyendo HAHV File 2");
            dict.Add(TagType.HAHV, TrainerFileSelector.ReadSqlite(Path.Combine(dataPath, EEGEmoProc2ChSettings.Instance.HAHVFileName.Value)));
            Console.WriteLine("Leyendo LALV File 3");
            dict.Add(TagType.LALV, TrainerFileSelector.ReadSqlite(Path.Combine(dataPath, EEGEmoProc2ChSettings.Instance.LALVFileName.Value)));
            Console.WriteLine("Leyendo LAHV File 4");
            dict.Add(TagType.LAHV, TrainerFileSelector.ReadSqlite(Path.Combine(dataPath, EEGEmoProc2ChSettings.Instance.LAHVFileName.Value)));
            Console.WriteLine("procesados, ahora a entrenar");
            for (int i = 0; i < numberExecutions; i++)
            {
                string instanceDir = Path.Combine(Path.Combine(assemblyPath, "salida"), "instance" + i);
                if (!Directory.Exists(instanceDir))
                {
                    Directory.CreateDirectory(instanceDir);
                }
                var lm = new LearningModel(instanceDir,
                    EEGEmoProc2ChSettings.Instance.IterationsMax.Value,
                    EEGEmoProc2ChSettings.Instance.NumberStars.Value,
                    EEGEmoProc2ChSettings.Instance.MinError.Value,
                    EEGEmoProc2ChSettings.Instance.MaxParalelism.Value);
                lm.Train(dict);
            }
            /*double[][] data = { new double[] { 6.4688626447614546, 7.5686374989548071, 6.7558982629920026, 5.1081076369050757, 6.8749737894577576, 8.2617846795147525, 6.555227567529851, 6.5508205610453212 }, new double[] { 7.3453648404168685, 7.8564486619637925, 0.0, 8.95531908176395, 0.0, 6.7566739579242387, 6.650149674112825, 5.4540382415448123 }, new double[] { 6.7569323892475532, 6.5558740217510838, 5.6893194126286462, 5.2002056470458937, 6.7570615798684939, 6.8748446155248777, 5.8575434681760017, 7.8552862465923488 }, new double[] { 7.8563195714065879, 7.3453648404168685, 4.8675344504555822, 4.8346663977148232, 8.2620428439669418, 7.1629141597283583, 5.3919378078597182, 4.4176215084244808 }, new double[] { 7.1629141597283583, 6.6515718735897273, 6.7543450656826023, 7.0037145986989024, 8.2620428439669418, 6.8748446155248777, 5.8090560300113507, 5.8588850822088414 }, new double[] { 7.1629141597283583, 6.8748446155248777, 6.1180971980413483, 8.2616555722910228, 7.3453648404168685, 7.8564486619637925, 7.3435555868863744, 6.464718097966216 }, new double[] { 7.8563195714065879, 6.7570615798684939, 6.6497614502321811, 6.1774253072944756, 8.2620428439669418, 6.8748446155248777, 5.6904893835128352, 5.9510331013136435 }, new double[] { 0.0, 7.3453648404168685, 7.8532163881560724, 6.6461307408496459, 8.95531908176395, 7.8563195714065879, 6.75680318193424, 6.0919750853218106 }, new double[] { 7.5686374989548071, 8.95531908176395, 5.5811601623590681, 6.648466281031574, 8.95531908176395, 0.0, 7.0069539925261539, 6.64678004935892 }, new double[] { 6.8749737894577576, 7.00850518208228, 5.51563754326907, 5.2690469842479235, 8.95531908176395, 7.5686374989548071, 7.3453648404168685, 7.3436849278941212 }, new double[] { 7.1629141597283583, 7.8564486619637925, 4.6050331903032413, 4.8797403561350858, 7.1629141597283583, 7.3453648404168685, 6.3880441418589378, 6.5501708306605639 }, new double[] { 7.3453648404168685, 7.8563195714065879, 6.1155024897874757, 6.4665334511618537, 7.3452357165223123, 7.1629141597283583, 6.7566739579242387, 5.4354407849650146 } };
            int[] outputs = { 2, 2, 2, 3, 3, 3, 0, 0, 0, 1, 1, 1 };

            string assemblyPath = System.IO.Path.GetDirectoryName(System.Reflection.Assembly.GetEntryAssembly().Location);
            string svmPath = Path.Combine(Path.Combine(Path.Combine(assemblyPath, "salida"), "instance0"), "emotionmodel.svm");
            MulticlassSupportVectorMachine<Gaussian> svm = Serializer.Load<MulticlassSupportVectorMachine<Gaussian>>(path: svmPath);

            int i = 0;
            foreach(var feature in data)
            {
                int val = svm.Decide(feature);
                Console.WriteLine(val);
                if (val.Equals(outputs[i]))
                {
                    Console.WriteLine("Dio bien");
                }
                else
                {
                    Console.WriteLine("Dio mal");
                }
                i++;
            }*/

        }
    }
}
