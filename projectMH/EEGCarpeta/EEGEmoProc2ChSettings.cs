using Config.Net;

namespace cl.uv.leikelen.Module.Processing.EEGEmotion2Channels
{
    public class EEGEmoProc2ChSettings : SettingsContainer
    {
        public readonly Option<int> Decimation = new Option<int>("Decimation", 0);
        public readonly Option<int> SamplingHz = new Option<int>("SamplingHz", 128);
        public readonly Option<int> m = new Option<int>("m", 2);
        public readonly Option<double> r = new Option<double>("r", 0.15);
        public readonly Option<int> secs = new Option<int>("Seconds", 9);
        public readonly Option<int> N = new Option<int>("N", 128);
        public readonly Option<int> shift = new Option<int>("shift", 1);
        
        public readonly Option<int> Executions = new Option<int>("Executions", 1);
        public readonly Option<int> IterationsMax = new Option<int>("IterationsMax", 100);
        public readonly Option<int> NumberStars = new Option<int>("NumberStars", 30);
        public readonly Option<double> MinError = new Option<double>("MinError", 0);
        public readonly Option<int> MaxParalelism = new Option<int>("MaxParalelism", 1);
        public readonly Option<bool> OnlyPaper = new Option<bool>("OnlyPaper", false);

        public readonly Option<string> LALVFileName = new Option<string>("LALVFileName", "LALV.db");
        public readonly Option<string> LAHVFileName = new Option<string>("LAHVFileName", "LAHV.db");
        public readonly Option<string> HALVFileName = new Option<string>("HALVFileName", "HALV.db");
        public readonly Option<string> HAHVFileName = new Option<string>("HAHVFileName", "HAHV.db");

        public readonly Option<int> MaxLines = new Option<int>("MaxLinesOrNegative", -1);

        private static EEGEmoProc2ChSettings _instance;

        public static EEGEmoProc2ChSettings Instance
        {
            get
            {
                if (ReferenceEquals(null, _instance)) _instance = new EEGEmoProc2ChSettings();
                return _instance;
            }
        }
        
        protected override void OnConfigure(IConfigConfiguration configuration)
        {
            configuration.UseJsonFile(@"config.json");
        }
    }
}
