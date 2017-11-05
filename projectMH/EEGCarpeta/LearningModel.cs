using Accord.IO;
using Accord.MachineLearning;
using Accord.MachineLearning.Performance;
using Accord.MachineLearning.VectorMachines;
using Accord.MachineLearning.VectorMachines.Learning;
using Accord.Statistics.Kernels;
using Accord.Math.Optimization.Losses;
using cl.uv.leikelen.Module.Processing.EEGEmotion2Channels.PreProcessing;
using System;
using System.Collections.Generic;
using System.IO;

namespace cl.uv.leikelen.Module.Processing.EEGEmotion2Channels
{
    public class LearningModel
    {
        private FilterButterworth _lowFilter;
        private FilterButterworth _highFilter;
        private MulticlassSupportVectorMachine<Gaussian> _svm;
        private string _directory;
        private int _iterations;
        private int _population;
        private double _minError;
        private int _paralelism;
        private List<Tuple<double[], int>> _originalInputsList;

        public LearningModel(string directory, int iterations, int population, double minError, int paralelism = 1)
        {
            this._directory = directory;
            _iterations = iterations;
            _population = population;
            _minError = minError;
            _paralelism = paralelism;
        }

        private Random _xrand;// = new Random(DateTime.Now.Second);
        
        public struct Star2
        {
            public MulticlassSupportVectorMachine<Gaussian> svm;
            public double error;
            public double Complexity;
            public double Gamma;
            public List<double[]> inputsList;
        }

        public MulticlassSupportVectorMachine<Gaussian> Train(Dictionary<TagType, List<List<double[]>>> allsignalsList)
        {
            int seed = DateTime.Now.Second+ DateTime.Now.Millisecond+ DateTime.Now.Minute;
            Console.WriteLine("semilla: "+seed);
            _xrand = new Random(seed);
            Console.WriteLine("a entrenar se ha dicho");
            List<double[]> inputsList = new List<double[]>();
            List<int> outputsList = new List<int>();
            _originalInputsList = new List<Tuple<double[], int>>();
            foreach (var tag in allsignalsList.Keys)
            {
                Console.WriteLine("empezando a procesar el tag:"+ tag);
                int i = 0;
                foreach (var signalList in allsignalsList[tag])
                {
                    double[] featureVector = PreProcess(signalList,
                        0.99,
                        EEGEmoProc2ChSettings.Instance.m.Value,
                        EEGEmoProc2ChSettings.Instance.r.Value,
                        EEGEmoProc2ChSettings.Instance.N.Value,
                        1,
                        0).ToArray();

                    inputsList.Add(featureVector.DeepClone());
                    outputsList.Add(tag.DeepClone().GetHashCode());
                    _originalInputsList.Add(new Tuple<double[], int>(featureVector.DeepClone(), tag.DeepClone().GetHashCode()));
                    //Console.WriteLine(_originalInputsList[i].Item1+","+ _originalInputsList[i].Item2);
                    i = i + 1;
                }
            }
            Console.WriteLine("procesado todo, ahora a buscar");
            bool onlyPaper = EEGEmoProc2ChSettings.Instance.OnlyPaper.Value;
            if (onlyPaper)
            {
                var res = TrainingPaper(inputsList, outputsList);
                WriteFiles(res.Item2, res.Item4, res.Item3, inputsList, outputsList, res.Item1);
                return res.Item1;
            }
            // Instantiate a new Grid Search algorithm for Kernel Support Vector Machines
            MulticlassSupportVectorMachine<Gaussian> svm = null;// Training(inputsList, outputsList).BestModel;
            Tuple<MulticlassSupportVectorMachine<Gaussian>, double, double, double> result;// Training(inputsList, outputsList);
            var fileBestByIt = File.AppendText(Path.Combine(_directory, "historial_best_by_it.txt"));
            var file_all_stars = File.AppendText(Path.Combine(_directory, "all_stars_all_its.txt"));
            Star2[] stars = new Star2[_population];
            Star2 best;
            try
            {
                result = Training(inputsList, outputsList);
                best = new Star2()
                {
                    svm = result.Item1.DeepClone(),
                    error = result.Item2,
                    Complexity = result.Item3,
                    Gamma = result.Item4,
                    inputsList = inputsList.DeepClone()

                };
            }
            catch (Exception ex)
            {
                Console.WriteLine("Error al entrenar el primero: " + ex.Message + "\nInner: " +ex.InnerException?.Message);
                best = new Star2()
                {
                    svm = null,
                    error = 1,
                    Complexity = -1,
                    Gamma = -1,
                    inputsList = inputsList.DeepClone()

                };
            }
            fileBestByIt.WriteLine("Iteration: Initial model, Seed: " + seed + ", Error: " + best.error + ", Gamma: " + best.Gamma + ", C: " + best.Complexity + "\n inputs: " + best.inputsList.ToJsonString(true));
            fileBestByIt.Flush();

            file_all_stars.WriteLine("Model: inicial, Seed: "+ seed+", Error: "+best.error+", Gamma: "+best.Gamma+", C: "+best.Complexity+"\n inputs: "+best.inputsList.ToJsonString(true));
            file_all_stars.Flush();
            //inicialization
            for (int iStar = 0; iStar < stars.Length; iStar++)
            {
                stars[iStar] = new Star2()
                {
                    svm = best.svm?.DeepClone(),
                    error = best.error,
                    Complexity = best.Complexity,
                    Gamma = best.Gamma,
                    inputsList = inputsList.DeepClone()
                };
                try
                {
                    for (int iInput = 0; iInput < stars[iStar].inputsList.Count; iInput++)
                    {
                        for (int jinput = 0; jinput < stars[iStar].inputsList[iInput].Length; jinput++)
                        {
                            stars[iStar].inputsList[iInput][jinput] = stars[iStar].inputsList[iInput][jinput]
                                + ((_xrand.NextDouble() * -1) * stars[iStar].error * stars[iStar].inputsList[iInput][jinput]);
                        }

                    }
                    var res = Training(stars[iStar].inputsList, outputsList);
                    stars[iStar].svm = res.Item1;
                    stars[iStar].error = res.Item2;
                    stars[iStar].Complexity = res.Item3;
                    stars[iStar].Gamma = res.Item4;
                    file_all_stars.WriteLine("Model: -1, Seed: "+seed+", Error: "+stars[iStar].error+", Gamma: "+stars[iStar].Gamma+", C: "+stars[iStar].Complexity+"\n inputs: "+stars[iStar].inputsList.ToJsonString(true));
                    file_all_stars.Flush();
                }
                catch(Exception ex)
                {
                    Console.WriteLine("Error al inicializar estrella "+iStar+": "+ex.Message+"\nInner: "+ex.InnerException?.Message);
                }
                
            }
            foreach (var star in stars)
            {
                if (star.error < best.error)
                {
                    best.svm = star.svm?.DeepClone();
                    best.error = star.error;
                    best.Complexity = star.Complexity;
                    best.Gamma = star.Gamma;
                    best.inputsList = star.inputsList.DeepClone();
                }
            }
            //cycle
            for (int i = 0; i < _iterations; i++)
            {
                Console.WriteLine("-----------------------------------------------------------------------------------------------");
                Console.WriteLine("Iteration: " + i + ", Seed" + seed + ", Error: " + best.error);
                Console.WriteLine("-----------------------------------------------------------------------------------------------");
                fileBestByIt.WriteLine("Iteration: "+i+", Seed: "+seed+", Error: "+best.error+", Gamma: "+best.Gamma+", C: "+best.Complexity+"\n inputs: "+best.inputsList.ToJsonString(true));
                fileBestByIt.Flush();
                
                if (best.error <= _minError)
                {
                    svm = best.svm;
                    WriteFiles(best.error, best.Gamma, best.Complexity, best.inputsList, outputsList, best.svm);
                    return svm;
                }
                //each star
                for(int iStar = 0; iStar < stars.Length; iStar++)
                {
                    Star2 prevStar = new Star2()
                    {
                        svm = stars[iStar].svm?.DeepClone(),
                        error = stars[iStar].error,
                        Complexity = stars[iStar].Complexity,
                        Gamma = stars[iStar].Gamma,
                        inputsList = stars[iStar].inputsList.DeepClone()
                    };
                    try
                    {
                        for (int iInput = 0; iInput < stars[iStar].inputsList.Count; iInput++)
                        {
                            for (int jinput = 0; jinput < stars[iStar].inputsList[iInput].Length; jinput++)
                            {
                                stars[iStar].inputsList[iInput][jinput] = stars[iStar].inputsList[iInput][jinput]
                                    + _xrand.NextDouble()
                                    * (best.inputsList[iInput][jinput] - stars[iStar].inputsList[iInput][jinput]);
                            }
                        }
                        var res = Training(stars[iStar].inputsList, outputsList);
                        stars[iStar].svm = res.Item1;
                        stars[iStar].error = res.Item2;
                        stars[iStar].Complexity = res.Item3;
                        stars[iStar].Gamma = res.Item4;
                        file_all_stars.WriteLine("it: "+i+", Model: "+iStar+" Seed: "+seed+", Error: "+stars[iStar].error+", Gamma: "+stars[iStar].Gamma+", C: "+stars[iStar].Complexity+"\n inputs: "+stars[iStar].inputsList.ToJsonString(true));
                        file_all_stars.Flush();
                    }
                    catch(Exception ex)
                    {
                        Console.WriteLine("Error en it "+i+" al cambiar estrella " + iStar + ": " + ex.Message+"\n Inner: "+ex.InnerException?.Message);
                        stars[iStar] = prevStar;
                    }
                }
                for (int iStar = 0; iStar < stars.Length; iStar++)
                {
                    if (stars[iStar].error < best.error)
                    {
                        best.svm = stars[iStar].svm?.DeepClone();
                        best.error = stars[iStar].error;
                        best.Complexity = stars[iStar].Complexity;
                        best.Gamma = stars[iStar].Gamma;
                        best.inputsList = stars[iStar].inputsList.DeepClone();
                    }
                    double sumError = 0;
                    for(int jStar = 0; jStar < stars.Length; jStar++)
                    {
                        sumError += stars[jStar].error;
                    }
                    if((best.error/sumError) < _xrand.NextDouble() * 0.1)
                    {
                        Console.WriteLine("Se alcanzó el horizonte de eventos en "+iStar);
                        for (int iInput = 0; iInput < stars[iStar].inputsList.Count; iInput++)
                        {
                            for (int jinput = 0; jinput < stars[iStar].inputsList[iInput].Length; jinput++)
                            {
                                stars[iStar].inputsList[iInput][jinput] = stars[iStar].inputsList[iInput][jinput]
                                    + ((_xrand.NextDouble() * -1) * stars[iStar].error * stars[iStar].inputsList[iInput][jinput]);
                            }

                        }
                    }
                }
                svm = best.svm;
            }
            fileBestByIt.Close();
            file_all_stars.Close();
            WriteFiles(best.error, best.Gamma, best.Complexity, inputsList, outputsList, best.svm);
            return svm;
        }

        private Tuple<MulticlassSupportVectorMachine<Gaussian>, double, double, double> Training(List<double[]> inputsList, List<int> outputsList)
        {
            /*var gridsearch = new GridSearch<MulticlassSupportVectorMachine<Gaussian>, double[], int>()
            {
                // Here we can specify the range of the parameters to be included in the search
                ParameterRanges = new GridSearchRangeCollection()
                {
                    new GridSearchRange("Complexity", new double[]{Math.Pow(2, -10), Math.Pow(2, -8),
                        Math.Pow(2, -6), Math.Pow(2, -4), Math.Pow(2, -2), Math.Pow(2, 0), Math.Pow(2, 2),
                        Math.Pow(2, 4), Math.Pow(2, 6), Math.Pow(2, 8), Math.Pow(2, 10)}),
                    new GridSearchRange("Gamma", new double[]{Math.Pow(2, -10), Math.Pow(2, -8),
                        Math.Pow(2, -6), Math.Pow(2, -4), Math.Pow(2, -2), Math.Pow(2, 0), Math.Pow(2, 2),
                        Math.Pow(2, 4), Math.Pow(2, 6), Math.Pow(2, 8), Math.Pow(2, 10)})
                },

                // Indicate how learning algorithms for the models should be created
                Learner = (p) => new MulticlassSupportVectorLearning<Gaussian>()
                {
                    // Configure the learning algorithm to use SMO to train the
                    //  underlying SVMs in each of the binary class subproblems.
                    Learner = (param) => new SequentialMinimalOptimization<Gaussian>()
                    {
                        // Estimate a suitable guess for the Gaussian kernel's parameters.
                        // This estimate can serve as a starting point for a grid search.
                        UseComplexityHeuristic = true,
                        UseKernelEstimation = true
                        //Complexity = p["Complexity"],
                        //Kernel = Gaussian.FromGamma(p["Gamma"])
                    }
                },
                // Define how the model should be learned, if needed
                Fit = (teacher, x, y, w) => teacher.Learn(x, y, w),

                // Define how the performance of the models should be measured
                Loss = (actual, expected, m) =>
                {
                    double totalError = 0;
                    foreach (var input in _originalInputsList)
                    {
                        if (!m.Decide(input.Item1).Equals(input.Item2))
                        {
                            totalError++;
                        }
                    }
                    return totalError / _originalInputsList.Count;
                }
            };*/
            var Learner = new MulticlassSupportVectorLearning<Gaussian>()
            {
                // Configure the learning algorithm to use SMO to train the
                //  underlying SVMs in each of the binary class subproblems.
                Learner = (param) => new SequentialMinimalOptimization<Gaussian>()
                {
                    // Estimate a suitable guess for the Gaussian kernel's parameters.
                    // This estimate can serve as a starting point for a grid search.
                    UseComplexityHeuristic = true,
                    UseKernelEstimation = true
                    //Complexity = p["Complexity"],
                    //Kernel = Gaussian.FromGamma(p["Gamma"])
                }
            };

            

            Learner.ParallelOptions.MaxDegreeOfParallelism = _paralelism;
            var model = Learner.Learn(inputsList.ToArray(), outputsList.ToArray());

            Console.WriteLine("y nos ponemos a aprender");
            // Search for the best model parameters
            //var result = gridsearch.Learn(inputsList.ToArray(), outputsList.ToArray());
            //Console.WriteLine("Error modelo: " + result.BestModelError);

            //var model = result.BestModel;
            double gamma = model.Kernel.Gamma;
            double error = 0;
            foreach (var input in _originalInputsList)
            {
                if (!model.Decide(input.Item1).Equals(input.Item2))
                {
                    error++;
                }
            }
            error = error / _originalInputsList.Count;

            return new Tuple<MulticlassSupportVectorMachine<Gaussian>, double, double, double>(model, error, gamma, 0);
        }

        private Tuple<MulticlassSupportVectorMachine<Gaussian>, double, double, double> TrainingPaper(List<double[]> inputsList, List<int> outputsList)
        {
            var gridsearch = GridSearch<double[], int>.CrossValidate(
                // Here we can specify the range of the parameters to be included in the search
                ranges: new
                {
                    Complexity = GridSearch.Values(Math.Pow(2, -12), Math.Pow(2, -11), Math.Pow(2, -10), Math.Pow(2, -8),
                        Math.Pow(2, -6), Math.Pow(2, -4), Math.Pow(2, -2), Math.Pow(2, 0), Math.Pow(2, 2),
                        Math.Pow(2, 4), Math.Pow(2, 6), Math.Pow(2, 8), Math.Pow(2, 10), Math.Pow(2, 11), Math.Pow(2, 12)),
                    Gamma = GridSearch.Values(Math.Pow(2, -12), Math.Pow(2, -11), Math.Pow(2, -10), Math.Pow(2, -8),
                        Math.Pow(2, -6), Math.Pow(2, -4), Math.Pow(2, -2), Math.Pow(2, 0), Math.Pow(2, 2),
                        Math.Pow(2, 4), Math.Pow(2, 6), Math.Pow(2, 8), Math.Pow(2, 10), Math.Pow(2, 11), Math.Pow(2, 12))
                },

                // Indicate how learning algorithms for the models should be created
                learner: (p, ss) => new MulticlassSupportVectorLearning<Gaussian>()
                {
                    
                    // Configure the learning algorithm to use SMO to train the
                    //  underlying SVMs in each of the binary class subproblems.
                    Learner = (param) => new SequentialMinimalOptimization<Gaussian>()
                    {
                        // Estimate a suitable guess for the Gaussian kernel's parameters.
                        // This estimate can serve as a starting point for a grid search.
                        //UseComplexityHeuristic = true,
                        //UseKernelEstimation = true
                        Complexity = p.Complexity,
                        Kernel = Gaussian.FromGamma(p.Gamma)
                    }
                },
                // Define how the model should be learned, if needed
                fit: (teacher, x, y, w) => teacher.Learn(x, y, w),

                // Define how the performance of the models should be measured
                /*loss: (actual, expected, m) =>
                {
                    double totalError = 0;
                    foreach (var input in _originalInputsList)
                    {
                        if (!m.Decide(input.Item1).Equals(input.Item2))
                        {
                            totalError++;
                        }
                    }
                    return totalError / _originalInputsList.Count;
                },*/
                loss: (actual, expected, m) => new HammingLoss(expected).Loss(actual),
                folds: 10
            );

            gridsearch.ParallelOptions.MaxDegreeOfParallelism = _paralelism;

            Console.WriteLine("y nos ponemos a aprender");
            // Search for the best model parameters
            var result = gridsearch.Learn(inputsList.ToArray(), outputsList.ToArray());
            Console.WriteLine("Error modelo: " + result.BestModelError);

            var model = CreateModel(inputsList, outputsList, result.BestParameters.Complexity, result.BestParameters.Gamma);

            double error = 0;
            Console.WriteLine("Largo: " + _originalInputsList.Count);
            foreach (var input in _originalInputsList)
            {
                if (!model.Decide(input.Item1).Equals(input.Item2))
                {
                    error++;
                }
            }
            error = error / (_originalInputsList.Count);
            Console.WriteLine("Error real: " + error);
            
            return new Tuple<MulticlassSupportVectorMachine<Gaussian>, double, double, double>(model, error, result.BestParameters.Gamma.Value, result.BestParameters.Complexity.Value);
        }

        private MulticlassSupportVectorMachine<Gaussian> CreateModel(List<double[]> inputsList,
            List<int> outputsList, double complexity, double gamma)
        {
            var teacher =  new MulticlassSupportVectorLearning<Gaussian>()
            {
                // Configure the learning algorithm to use SMO to train the
                //  underlying SVMs in each of the binary class subproblems.
                Learner = (param) => new SequentialMinimalOptimization<Gaussian>()
                {
                    // Estimate a suitable guess for the Gaussian kernel's parameters.
                    // This estimate can serve as a starting point for a grid search.

                    Complexity = complexity,
                    Kernel = Gaussian.FromGamma(gamma)
                }
            };
            return teacher.Learn(inputsList.ToArray(), outputsList.ToArray());
        }

        private void WriteFiles(double error, double gamma, double complexity,
            List<double[]> inputsList, List<int> outputsList, MulticlassSupportVectorMachine<Gaussian> svm)
        {
            // Get the best SVM found during the parameter search
            _svm = svm;
            
            Console.WriteLine("error: "+error+", Gamma: "+gamma+", C: "+complexity);
            string outInternalPath = Path.Combine(_directory, "result.txt");
            var file_emotrain = File.CreateText(outInternalPath);
            string executionType = EEGEmoProc2ChSettings.Instance.OnlyPaper.Value ? "Only paper" : "With Metaheuristic";
            file_emotrain.WriteLine(executionType);
            file_emotrain.WriteLine("error: " + error + ", Gamma: " + gamma + ", C: " + complexity);
            file_emotrain.WriteLine("Gamma kernel: " + svm?.Kernel.Gamma + ", Classes: " + svm?.NumberOfClasses);
            file_emotrain.WriteLine("Inner binary svm models (classes*(classes-1))/2:" + svm?.Count);
            file_emotrain.Flush();
            file_emotrain.Close();

            string internalPath = Path.Combine(_directory, "emotionmodel.svm");
            try
            {
                Serializer.Save<MulticlassSupportVectorMachine<Gaussian>>(obj: svm, path: internalPath);
                Console.WriteLine("guardado");
            }catch(Exception ex)
            {
                Console.WriteLine("Error al guardar svm file: "+ex.Message+"\n Inner: "+ex.InnerException?.Message);
            }
            

            var file_features = File.CreateText(Path.Combine(_directory, "features.json"));
            file_features.WriteLine(inputsList.ToJsonString());
            file_features.Flush();
            file_features.Close();

            var file_outputs = File.CreateText(Path.Combine(_directory, "outputs.json"));
            file_outputs.WriteLine(outputsList.ToJsonString());
            file_outputs.Flush();
            file_outputs.Close();
        }

        private List<double> MH(double[][] imfsF3, double[][] imfsC4)
        {
            List<double> features = new List<double>();
            int n = 20;
            double[][] x = new double[n][], v = new double[n][];

            foreach(var imfSignal in imfsF3)
            {
                features.Add(MHEntropy(imfSignal));
            }
            foreach(var imfSignal in imfsC4)
            {
                features.Add(MHEntropy(imfSignal));
            }
            return features;
        }

        class Star
        {
            public double position;
        };

        private double MHEntropy(double[] signal)
        {
            int starNumber = 30;
            int iterations = 30;
            double realEntropy = RealEntropy(signal);

            Star[] stars = new Star[starNumber];
            for(int i = 0; i < starNumber; i++)
            {
                stars[i] = new Star
                {
                    position = _xrand.NextDouble() * 8
                };
            }

            for(int i = 0; i < iterations; i++)
            {
                for(int iStar = 0; iStar<stars.Length;iStar++)
                {
                    stars[iStar].position = stars[iStar].position + _xrand.NextDouble() * (BestStar(stars, realEntropy).position - stars[iStar].position);
                }
            }
            
            double entropyResult = BestStar(stars, realEntropy).position;
            Console.WriteLine("entropy: " + entropyResult);
            return entropyResult;
        }

        private Star BestStar(Star[] stars, double realEntropy)
        {
            double bestDiference = double.PositiveInfinity;
            Star bestStar = null;
            foreach(var star in stars)
            {
                if (ReferenceEquals(null, bestStar))
                {
                    bestStar = star;
                }
                else
                {
                    double diference = Math.Abs(star.position - realEntropy);
                    if (diference < bestDiference)
                    {
                        bestStar = star;
                    }
                }
            }
            return bestStar;
        }

        private double RealEntropy(double[] signal)
        {
            double[] integerSignal = new double[signal.Length];
            Dictionary<double, int> histogram = new Dictionary<double, int>();
            double entropy = 0;
            for (int i = 0; i < signal.Length;i++)
            {
                integerSignal[i] = Math.Abs(signal[i]);
            }

            foreach(var val in integerSignal)
            {
                if (histogram.ContainsKey(val))
                {
                    histogram[val]++;
                }
                else
                {
                    histogram.Add(val, 0);
                }
            }
            foreach(var key in histogram.Keys)
            {
                double p = histogram[key] / histogram.Keys.Count;
                entropy += p * (Math.Log10(p) / Math.Log10(2));
            }
            return entropy;
        }

        private List<double> PreProcess(List<double[]> signalsList, double Q, int m, double r, int N, int iterations, int locality)
        {
            double[] f3 = new double[signalsList.Count];
            bool first = true;


            for (int i = 0; i < f3.Length; i++)
            {
                if (first)
                {
                    //f3[i] = BetaBandpass(signalsList[i][0], true);
                    f3[i] = signalsList[i][0];
                    first = false;
                }
                else
                {
                    //f3[i] = BetaBandpass(signalsList[i][0], false);
                    f3[i] = signalsList[i][0];
                }
            }
            
            //f3 = FilterRLC.LCHP(f3, EEGEmoProc2ChSettings.Instance.SamplingHz.Value, 5, Q);
            //f3 = FilterRLC.LCLP(f3, EEGEmoProc2ChSettings.Instance.SamplingHz.Value, 50, Q);
            for(int i=0; i < f3.Length; i++)
            {
                if (double.IsInfinity(f3[i]) || double.IsNaN(f3[i]))
                {
                    Console.WriteLine("F3 es NaN o Infinityyyyyy");
                    f3[i] = f3[i - 1] * (1+(_xrand.Next(-1,1)/10));
                }
            }


            double[] c4 = new double[signalsList.Count];
            first = true;
            for (int i = 0; i < c4.Length; i++)
            {
                if (first)
                {
                    c4[i] = BetaBandpass(signalsList[i][1], true);
                    //c4[i] = signalsList[i][1];
                    first = false;
                }
                else
                {
                    c4[i] = BetaBandpass(signalsList[i][1], false);
                    //c4[i] = signalsList[i][1];
                }
            }
            //c4 = FilterRLC.LCHP(c4, EEGEmoProc2ChSettings.Instance.SamplingHz.Value, 5, Q);
            //c4 = FilterRLC.LCLP(c4, EEGEmoProc2ChSettings.Instance.SamplingHz.Value, 50, Q);
            for (int i = 0; i < c4.Length; i++)
            {
                if (double.IsInfinity(c4[i]) || double.IsNaN(c4[i]))
                {
                    Console.WriteLine("C4 es NaN o Infinityyyyyy");
                    c4[i] = c4[i - 1] * (1 + (_xrand.Next(-1, 1) / 10));
                }
            }
            var emdF3 = new Emd();
            var imfsF3 = emdF3.GetImfs(f3, 4, iterations, locality);

            var emdC4 = new Emd();
            var imfsC4 = emdC4.GetImfs(c4, 4, iterations, locality);
            
            //return MH(imfsF3, imfsC4);
            return CalcEntropy(imfsF3, imfsC4, N, m, r);
        }

        private List<double> CalcEntropy(double[][] imfsF3, double[][] imfsC4, int N, int m, double r) { 

            List<double> features = new List<double>();
            foreach (var imfF3 in imfsF3)
            {
                features.Add(SampleEntropy.CalcSampleEntropy(imfF3, N, m, r,
                    EEGEmoProc2ChSettings.Instance.shift.Value));
            }

            foreach (var imfC4 in imfsC4)
            {
                features.Add(SampleEntropy.CalcSampleEntropy(imfC4, N, m, r,
                    EEGEmoProc2ChSettings.Instance.shift.Value));
            }
            return features;
        }

        private double BetaBandpass(double signal, bool newFilters)
        {
            if (newFilters || ReferenceEquals(null, _lowFilter) || ReferenceEquals(null, _highFilter))
            {
                _lowFilter = new FilterButterworth(45.0f,
                    EEGEmoProc2ChSettings.Instance.SamplingHz,
                    FilterButterworth.PassType.Lowpass, 0.1f);

                _highFilter = new FilterButterworth(4.0f,
                    EEGEmoProc2ChSettings.Instance.SamplingHz,
                    FilterButterworth.PassType.Highpass, 0.1f);
            }
            return _highFilter.Update(_lowFilter.Update(signal));
        }
    }
}
