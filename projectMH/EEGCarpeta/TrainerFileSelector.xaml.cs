using Microsoft.Win32;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.IO;
using Microsoft.Data.Sqlite;

namespace cl.uv.leikelen.Module.Processing.EEGEmotion2Channels.View
{
    /// <summary>
    /// Lógica de interacción para TrainerFileSelector.xaml
    /// </summary>
    public partial class TrainerFileSelector
    {
        public TrainerFileSelector()
        {
        }

        public static List<List<double[]>> ReadSqlite(string fileName)
        {
            string cs = "Filename=" + fileName;
            var result = new List<List<double[]>>();
            Console.WriteLine("procesando: " + fileName);
            using (SqliteConnection con = new SqliteConnection(cs))
            {
                Console.WriteLine("conecta2");
                con.Open();
                Console.WriteLine("abierto");

                string stm = "SELECT * FROM data ORDER BY SESSIONID, id";

                using (SqliteCommand cmd = new SqliteCommand(stm, con))
                {
                    Console.WriteLine("comman2");
                    using (SqliteDataReader rdr = cmd.ExecuteReader())
                    {
                        Console.WriteLine("ejecuta3");

                        var frame = new List<double[]>();
                        int secStart = 0;
                        int i = 0;
                        int lastTime = 0;
                        double lastF3 = 0;
                        double lastC4 = 0;
                        int lastSessionId = 0;

                        while (rdr.Read())
                        {
                            if (EEGEmoProc2ChSettings.Instance.MaxLines.Value > 0 
                                && i > EEGEmoProc2ChSettings.Instance.MaxLines.Value)
                                break;
                            if (secStart == 0)
                            {
                                secStart = rdr.GetInt32(1);
                                lastTime = secStart;
                                lastSessionId = rdr.GetInt32(5);
                            }
                            if (!rdr.GetInt32(5).Equals(lastSessionId))
                            {
                                frame = new List<double[]>();
                                secStart = rdr.GetInt32(1);
                                Console.WriteLine("nuevo archivo con segundo: "+secStart+" y el anterior: "+lastTime+", en i: "+i);
                                lastTime = secStart;
                            }
                            double[] values = new double[2];
                            bool added = false;
                            switch (EEGEmoProc2ChSettings.Instance.SamplingHz.Value)
                            {
                                case 128:
                                    if (rdr.GetInt32(2) % 2 != 0)
                                    {
                                        values[0] = (lastF3 + rdr.GetDouble(3)) / 2;
                                        values[1] = (lastC4 + rdr.GetDouble(4)) / 2;
                                        added = true;
                                    }
                                    break;
                                case 256:
                                    values[0] = rdr.GetDouble(3);
                                    values[1] = rdr.GetDouble(4);
                                    added = true;
                                    break;
                            }
                            if (rdr.GetInt32(1) < secStart + EEGEmoProc2ChSettings.Instance.secs)
                            {
                                if (added)
                                    frame.Add(values);
                            }
                            else
                            {
                                result.Add(frame);
                                frame = new List<double[]>();
                                secStart = rdr.GetInt32(1);
                                Console.WriteLine("frame añadido en el segundo "+secStart+", en i: "+i);
                            }
                            i++;
                            lastSessionId = rdr.GetInt32(5);
                            lastTime = secStart;
                            lastF3 = rdr.GetDouble(3);
                            lastC4 = rdr.GetDouble(4);


                        }
                    }
                }

                con.Close();
                return result;
            }
        }
    }
}
