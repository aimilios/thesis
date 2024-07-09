import pandas as pd
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import random as r
from skfuzzy import control as ctrl
import skfuzzy as fuzz
from scipy.interpolate import interp1d
import pickle
import random
import csv

class Sensor():
    
    def __init__(self,csv_row):
        
        self.Sensor_Id = int(csv_row['Sensor_Id'])      # Μοναδικο αναγνωριστικο αισθητηρα
        self.Sensor_Name = str(csv_row['Sensor_Name'])  # Ονομα αισθητηρα
        self.Upper_Node = int(csv_row['Upper_Node'])    # Ανω Κομβος σωληνα που ανοικει ο αισθητηρας
        self.Bottom_Node = int(csv_row['Bottom_Node'])  # Κατω Κομβος σωληνα που ανοικει ο αισθητηρας
        
        self.Index = int(csv_row['Index'])              # Θεση αισθητητρα στον σωληνα
        self.Dist = float(csv_row['Dist'])              # Αποσταση αισθητηρα απο Ανω Κομβο
        
        self.QMin = 0                                   # Ελαχιστη ροη σωληνα που ανοικει ο αισθητηρας
        self.QRush = float(csv_row['QRush'])            # Ροη αιχμης σωληνα που ανοικει ο αισθητηρας
        self.QMax = float(csv_row['QMax'])              # Μεγιστη ροη σωληνα που ανοικει ο αισθητηρας
        
        self.parameters_dict = None                     # Παραμετροι συναρτηση συμμετοχης 'Τιμης Ροης'
        self.input_fuzzy_variable_name = None           # Ονομα συναρτηση συμμετοχης 'Τιμης Ροης'
        self.input_fuzzy_variable = None                # ασαφης μεταβλητη 'Τιμης Ροης' απο skfuzzy
        
    def __str__(self):
        # Εμφανιση βασικων στοιχειων του αισθητηρα
        return f'({self.Sensor_Id})[{self.Sensor_Name}][{self.Upper_Node}->{self.Bottom_Node}/{self.Index}D{self.Dist}({self.QMin},{self.QRush},{self.QMax})'
        
    def round_params(self,params):
        # Στρογγυλοποιηση των παραμετρων την συναρτησης συμμετοχης
        return [round(x,4) for x in params]

    def calculate_membership_params(self):
        # Υπολογισμος παραμετρων οπως φαινεται στο διαγραμμα 4.6
        QMin = self.QMin
        QRush = self.QRush
        QMax = self.QMax
        median_flow = (QMin+QRush)*0.5
        quarter = abs(QRush-QMin)/4.0
        eighth = abs(QRush-QMin)/8.0
            
        parameters_dict = {}
        parameters_dict['LOW'] = self.round_params([QMin, QMin, QMin+quarter, QMin+quarter+eighth])
        parameters_dict['MEDIUM'] = [median_flow, eighth]
        parameters_dict['HIGH'] = self.round_params([QRush-quarter-eighth, QRush-quarter, QRush, QRush+eighth])
        parameters_dict['OVER'] = self.round_params([QRush,QRush+eighth,QMax,QMax])
        
        self.parameters_dict = parameters_dict
        
        
    def define_input_membership_function(self):
        # Ορισμος ασαφους μεταβλητης με βαση της παραμετρους που υπολογιστικων
        QMin = self.QMin
        QMax = self.QMax
        
        step = 0.000001
        input_variable_name = f'{self.Sensor_Id}-{self.Sensor_Name}'        
        
        F = ctrl.Antecedent(np.arange(QMin-step, QMax+step, step), input_variable_name)
        
        F['LOW'] = fuzz.trapmf(F.universe,self.parameters_dict['LOW'] )
        F['MEDIUM'] = fuzz.gaussmf(F.universe,self.parameters_dict['MEDIUM'][0],self.parameters_dict['MEDIUM'][1] )
        F['HIGH'] = fuzz.trapmf(F.universe, self.parameters_dict['HIGH'])
        F['OVER'] = fuzz.trapmf(F.universe, self.parameters_dict['OVER'])       
        
        self.input_fuzzy_variable = F
        self.input_fuzzy_variable_name = input_variable_name
        
    def compute_membership(self,value):
        # Υπολογιζει ολες της συναρτησεις συμμετοχης επιστρεφοντας ενα dictionary
        F = self.input_fuzzy_variable
        membership_values = {}
        for linguistic_term in F.terms.keys():
            membership_values[linguistic_term] = fuzz.interp_membership(F.universe, F[linguistic_term].mf, value)
        return membership_values        
        
        
class Pair():
    
    def __init__(self,Sensor_A,Sensor_B):
        self.Sensor_A = Sensor_A
        self.Sensor_B = Sensor_B
        self.Pair_Name = f'({self.Sensor_A.Sensor_Id}){self.Sensor_A.Sensor_Name}-({self.Sensor_B.Sensor_Id}){self.Sensor_B.Sensor_Name}'
        self.Pipe_Index = 0
        self.Pair_index = 0
        
        self.parameters_input = None
        self.input_fuzzy_variable = None
        self.input_fuzzy_variable_name = None
        
        self.parameters_output = None
        self.output_fuzzy_variable = None
        self.output_fuzzy_variable_name = None
        
        self.system_ctrl = None
        self.system_simulation = None
        
        self.result_matrix = None
        self.result_steps = 30        
        
    def round_params(self,params):
        return [round(x,7) for x in params]        
        
    def map_values(self,value, leftMin, leftMax, rightMin, rightMax):
        leftSpan = leftMax - leftMin
        rightSpan = rightMax - rightMin
        valueScaled = float(value - leftMin) / float(leftSpan)   
        return rightMin + (valueScaled * rightSpan)        
    
    def overlap_membership_param(self,param,overlap):
        h_overl = overlap*0.5
        return [param[0]-h_overl,param[1]+h_overl,param[2]-h_overl,param[3]+h_overl]
        
    def calculate_input_membership_params(self):
        QMax = self.Sensor_A.QMax
        
        parameters_dict = {}
        parameters_dict['ZERO'] = [0,0,3,3]
        parameters_dict['LOW'] = [3,3,6,6]
        parameters_dict['MEDIUM'] = [6,6, 14,14]
        parameters_dict['HIGH'] = [14,14, 20, 20]
        parameters_dict['OVER'] = [20,20, 100,100]
        
        mp = interp1d([0.0,100.0],[0.0,QMax])
               
        for key, parameters in parameters_dict.items():
            parameters_dict[key] = [mp(x) for x in parameters]
            parameters_dict[key] = self.overlap_membership_param(parameters_dict[key],overlap=0.00005)
            parameters_dict[key] = self.round_params(parameters_dict[key])
            
            self.swap_trap_vars(parameters_dict[key],1,2)
            
        self.parameters_input = parameters_dict
                    
    def define_input_membership_function(self):
        QMin = self.Sensor_A.QMin
        QMax = self.Sensor_A.QMax
        step = 0.000001 *2
        
        universe_values = np.arange(QMin-step, QMax+step, step)
        input_fuzzy_variable_name = f'Diff_{self.Pair_Name}'
        F = ctrl.Antecedent(universe_values,input_fuzzy_variable_name )

        F['ZERO'] = fuzz.trapmf(F.universe,self.parameters_input['ZERO'] )
        F['LOW'] = fuzz.trapmf(F.universe,self.parameters_input['LOW'])
        F['MEDIUM'] = fuzz.trapmf(F.universe,self.parameters_input['MEDIUM'])
        F['HIGH'] = fuzz.trapmf(F.universe, self.parameters_input['HIGH'])
        F['OVER'] = fuzz.trapmf(F.universe, self.parameters_input['OVER'])
    
        self.input_fuzzy_variable = F
        self.input_fuzzy_variable_name = input_fuzzy_variable_name
                
        
    def swap_trap_vars(self,parameters_dict,index_1,index_2):
        if (parameters_dict[index_1] >= parameters_dict[index_2]):
            temp = parameters_dict[index_1]
            parameters_dict[index_1] = parameters_dict[index_2]
            parameters_dict[index_2] = temp        
        
    def calculate_output_membership_params(self):
        QMin = self.Sensor_A.QMin
        QMax = self.Sensor_A.QMax
        parameters_dict = {}
        parameters_dict['LEAK_HIGH'] = [-5,-5,-4.5,-4.5]
        parameters_dict['LEAK_MEDIUM'] = [-4.5,-4.5,-3.75,-3.75]
        parameters_dict['LEAK_LOW'] = [-3.75,-3.75,-2.5,-2.5]
        parameters_dict['ZERO'] = [-2.5,-2.5,2.5,2.5]
        parameters_dict['INFLOW_LOW'] = [2.5,2.5,3.75,3.75]
        parameters_dict['INFLOW_MEDIUM'] = [3.75,3.75,4.5,4.5]
        parameters_dict['INFLOW_HIGH'] = [4.5,4.5,5,5]       
        mp = interp1d([-5.0,5.0],[0.0,QMax])
                
        for key, parameters in parameters_dict.items():
            new_parameters = [mp(x) for x in parameters]
            new_parameters = self.overlap_membership_param(new_parameters,overlap=0.0001) 
            new_parameters = self.round_params(new_parameters)                       
            self.swap_trap_vars(new_parameters,1,2)
            
            parameters_dict[key] = new_parameters
        
        self.parameters_output = parameters_dict      
        
        
    def define_output_membership_function(self):
        QMin = self.Sensor_A.input_fuzzy_variable.universe.min()
        QMax = self.Sensor_A.input_fuzzy_variable.universe.max()
        step = 0.000001*6
        universe_values = np.arange(QMin-step, QMax+step, step)
        
        output_fuzzy_variable_name = f'LOI_{self.Pair_Name}'
        
        F = ctrl.Consequent(universe_values, output_fuzzy_variable_name)
        F['LEAK_HIGH'] = fuzz.trapmf(F.universe, self.parameters_output['LEAK_HIGH'])
        F['LEAK_MEDIUM'] = fuzz.trapmf(F.universe, self.parameters_output['LEAK_MEDIUM'])
        F['LEAK_LOW'] = fuzz.trapmf(F.universe, self.parameters_output['LEAK_LOW'])
        F['ZERO'] = fuzz.trapmf(F.universe, self.parameters_output['ZERO'])
        F['INFLOW_LOW'] = fuzz.trapmf(F.universe, self.parameters_output['INFLOW_LOW'])
        F['INFLOW_MEDIUM'] = fuzz.trapmf(F.universe, self.parameters_output['INFLOW_MEDIUM'])
        F['INFLOW_HIGH'] = fuzz.trapmf(F.universe, self.parameters_output['INFLOW_HIGH'])
        
        self.output_fuzzy_variable = F  
        self.output_fuzzy_variable_name = output_fuzzy_variable_name
        
        
    def define_system_rules(self):
        I1 = self.Sensor_A.input_fuzzy_variable
        I2 = self.Sensor_B.input_fuzzy_variable
        D = self.input_fuzzy_variable
        O1 = self.output_fuzzy_variable
        
        rules = []
        # Κανονες Εισροης Χωρις Διαφορα
        rules.append(ctrl.Rule(I1['LOW'] & I2['MEDIUM'], O1['INFLOW_MEDIUM']))
        rules.append(ctrl.Rule(I1['LOW'] & I2['HIGH'], O1['INFLOW_HIGH']))
        rules.append(ctrl.Rule(I1['LOW'] & I2['OVER'], O1['INFLOW_HIGH']))
        rules.append(ctrl.Rule(I1['MEDIUM'] & I2['HIGH'], O1['INFLOW_MEDIUM']))
        rules.append(ctrl.Rule(I1['MEDIUM'] & I2['OVER'], O1['INFLOW_HIGH']))  
        rules.append(ctrl.Rule(I1['HIGH'] & I2['OVER'], O1['INFLOW_MEDIUM']))
        
        # Κανονες Διαροης Χωρις Διαφορα
        rules.append(ctrl.Rule(I2['LOW'] & I1['MEDIUM'], O1['LEAK_MEDIUM']))
        rules.append(ctrl.Rule(I2['LOW'] & I1['HIGH'], O1['LEAK_HIGH']))
        rules.append(ctrl.Rule(I2['LOW'] & I1['OVER'], O1['LEAK_HIGH'])) 
        rules.append(ctrl.Rule(I2['MEDIUM'] & I1['HIGH'], O1['LEAK_MEDIUM']))
        rules.append(ctrl.Rule(I2['MEDIUM'] & I1['OVER'], O1['LEAK_HIGH']))
        rules.append(ctrl.Rule(I2['HIGH'] & I1['OVER'], O1['LEAK_MEDIUM']))
        
        # Κανονες που λαμβανουν υποψη την διαφορα
        rules.append(ctrl.Rule(I1['LOW'] & I2['LOW'] & D['ZERO'], O1['ZERO']))
        rules.append(ctrl.Rule(I1['LOW'] & I2['LOW'] & D['LOW'], O1['INFLOW_LOW']))
        rules.append(ctrl.Rule(I1['LOW'] & I2['LOW'] & D['LOW'], O1['LEAK_LOW']))
        rules.append(ctrl.Rule(I1['LOW'] & I2['LOW'] & D['MEDIUM'], O1['INFLOW_MEDIUM']))
        rules.append(ctrl.Rule(I1['LOW'] & I2['LOW'] & D['MEDIUM'], O1['LEAK_MEDIUM']))
        rules.append(ctrl.Rule(I1['LOW'] & I2['LOW'] & D['HIGH'], O1['INFLOW_HIGH']))
        rules.append(ctrl.Rule(I1['LOW'] & I2['LOW'] & D['HIGH'], O1['LEAK_HIGH']))
        
        rules.append(ctrl.Rule(I1['MEDIUM'] & I2['MEDIUM'] & D['ZERO'], O1['ZERO']))
        rules.append(ctrl.Rule(I1['MEDIUM'] & I2['MEDIUM'] & D['LOW'], O1['INFLOW_LOW']))
        rules.append(ctrl.Rule(I1['MEDIUM'] & I2['MEDIUM'] & D['LOW'], O1['LEAK_LOW']))
        rules.append(ctrl.Rule(I1['MEDIUM'] & I2['MEDIUM'] & D['MEDIUM'], O1['INFLOW_MEDIUM']))
        rules.append(ctrl.Rule(I1['MEDIUM'] & I2['MEDIUM'] & D['MEDIUM'], O1['LEAK_MEDIUM']))
        rules.append(ctrl.Rule(I1['MEDIUM'] & I2['MEDIUM'] & D['HIGH'], O1['INFLOW_HIGH']))
        rules.append(ctrl.Rule(I1['MEDIUM'] & I2['MEDIUM'] & D['HIGH'], O1['LEAK_HIGH']))
        
        rules.append(ctrl.Rule(I1['HIGH'] & I2['HIGH'] & D['ZERO'], O1['ZERO']))
        rules.append(ctrl.Rule(I1['HIGH'] & I2['HIGH'] & D['LOW'], O1['INFLOW_LOW']))
        rules.append(ctrl.Rule(I1['HIGH'] & I2['HIGH'] & D['LOW'], O1['LEAK_LOW']))
        rules.append(ctrl.Rule(I1['HIGH'] & I2['HIGH'] & D['MEDIUM'], O1['INFLOW_MEDIUM']))
        rules.append(ctrl.Rule(I1['HIGH'] & I2['HIGH'] & D['MEDIUM'], O1['LEAK_MEDIUM']))
        rules.append(ctrl.Rule(I1['HIGH'] & I2['HIGH'] & D['HIGH'], O1['INFLOW_HIGH']))
        rules.append(ctrl.Rule(I1['HIGH'] & I2['HIGH'] & D['HIGH'], O1['LEAK_HIGH']))
        
        rules.append(ctrl.Rule(I1['OVER'] & I2['OVER'] & D['ZERO'], O1['ZERO']))
        rules.append(ctrl.Rule(I1['OVER'] & I2['OVER'] & D['LOW'], O1['INFLOW_HIGH']))
        rules.append(ctrl.Rule(I1['OVER'] & I2['OVER'] & D['LOW'], O1['LEAK_HIGH']))
        rules.append(ctrl.Rule(I1['OVER'] & I2['OVER'] & D['MEDIUM'], O1['INFLOW_HIGH']))
        rules.append(ctrl.Rule(I1['OVER'] & I2['OVER'] & D['MEDIUM'], O1['LEAK_HIGH']))
        rules.append(ctrl.Rule(I1['OVER'] & I2['OVER'] & D['HIGH'], O1['INFLOW_HIGH']))
        rules.append(ctrl.Rule(I1['OVER'] & I2['OVER'] & D['HIGH'], O1['LEAK_HIGH']))
        
        system_ctrl = ctrl.ControlSystem(rules)
        system_simulation = ctrl.ControlSystemSimulation(system_ctrl)
        
        self.system_ctrl = system_ctrl
        self.system_simulation = system_simulation
        
    def compute_system(self,s1_v,s2_v,s_diff):
        
        self.system_simulation.input[self.Sensor_A.input_fuzzy_variable_name] = s1_v
        self.system_simulation.input[self.Sensor_B.input_fuzzy_variable_name] = s2_v
        self.system_simulation.input[self.input_fuzzy_variable_name] = s_diff
    
        self.system_simulation.compute()
    
        system_result = self.system_simulation.output[self.output_fuzzy_variable_name]   
        
        return system_result
    
    def use_lookup_table(self,s1_index,s2_index):
        if (self.result_matrix is None):
            raise ValueError('Result Matrix empty,turn off lookup mode and compute lookup tables')
            
        return self.result_matrix[s1_index][s2_index]

    
    def compute_results(self,s1_v,s2_v,s_diff,mode_lookup = False):
        if (mode_lookup is True):
            to_res_index = interp1d([0.0,self.Sensor_A.QMax],[0,self.result_steps-1])
            s1_index = int(np.round(to_res_index(s1_v)))
            s2_index = int(np.round(to_res_index(s2_v)))  
            
            return self.use_lookup_table(s1_index,s2_index)
            
        else:
            return self.compute_system(s1_v,s2_v,s_diff)
        
    def compute_membership_input(self,value):
        F = self.input_fuzzy_variable
        membership_values = {}
        for linguistic_term in F.terms.keys():
            membership_values[linguistic_term] = fuzz.interp_membership(F.universe, F[linguistic_term].mf, value)
    
        return membership_values 
        
        
    def compute_membership_output(self,value):
        F = self.output_fuzzy_variable
        membership_values = {}
        for linguistic_term in F.terms.keys():
            membership_values[linguistic_term] = fuzz.interp_membership(F.universe, F[linguistic_term].mf, value)
    
        return membership_values         
        
    def pair_matrix(self):
        steps = 20
        s1_values = np.linspace(0.0, self.Sensor_A.QMax, steps)
        s2_values = np.linspace(0.0, self.Sensor_B.QMax, steps)
        result_matrix = np.zeros((steps, steps))
        for i, s1_v in enumerate(s1_values):
            for j, s2_v in enumerate(s2_values):      
                s_diff = abs(s1_v-s2_v)
                system_result = self.compute_system(s1_v,s2_v,s_diff)
                result_matrix[i, j] = system_result       
                
        return result_matrix,steps
    
    def vis_pair_matrix(self):
        s1 = self.Sensor_A
        s2 = self.Sensor_B
        
        result_matrix,steps = self.pair_matrix()
        s1_values = np.linspace(0.0, s1.QMax, steps)
        s2_values = np.linspace(0.0, s2.QMax, steps)
        
        plt.scatter(s2.QMax, s1.QMax, color='red', marker='x', label='')
        plt.plot([s2.QMax, s1.QMax], [s2.QMax, s1_values[0]], color='blue', linestyle='--', label='')
        plt.plot([s2.QMax, s2_values[0]], [s1.QMax, s1.QMax], color='green', linestyle='--', label='')
        plt.plot([0, s2_values[-1]], [0, s1_values[-1]], color='purple', linestyle='--', label='')

        plt.imshow(result_matrix, cmap='RdBu_r', extent=[s2_values[0], s2_values[-1], s1_values[0], s1_values[-1]], origin='lower')

        plt.colorbar(label='Inflow Result')
        plt.xlabel(f'{s2.Sensor_Name}')
        plt.ylabel(f'{s1.Sensor_Name}')
        plt.title('Inflow Result Matrix')
        plt.legend()
        plt.show()
            
                  
    def pprint_result(self,s1_v,s2_v,s_diff,mode_lookup=False):       
        s1_mems_vals = self.Sensor_A.compute_membership(s1_v)
        s2_mems_vals = self.Sensor_B.compute_membership(s2_v)
        diff_mems_vals = self.compute_membership_input(s_diff)
        
        s1_mems_max = max(s1_mems_vals, key=lambda k: s1_mems_vals[k])
        s2_mems_max = max(s2_mems_vals, key=lambda k: s2_mems_vals[k])
        diff_mems_max = max(diff_mems_vals, key=lambda k: diff_mems_vals[k])
        
        res = self.compute_results(s1_v,s2_v,s_diff,mode_lookup)
        res_mems_vals = self.compute_membership_output(res)
        res_mems_max = max(res_mems_vals, key=lambda k: res_mems_vals[k])
        
        pprinted_result = [s1_v,s1_mems_vals,s1_mems_max, 
                           s2_v,s2_mems_vals,s2_mems_max,
                           s_diff,diff_mems_vals,diff_mems_max,
                           res,res_mems_vals,res_mems_max]
        
        return pprinted_result
    
class sewage_system_main():  
    
    def __init__(self):
        # Directories Αρχειων
        self.network_data_path = r'kokkinia_data.csv'
        self.sensors_data_path = r'sensors_data.csv'
        self.pipe_sensors_path = r'pipe_sensors.csv'
        self.sensor_values_path = r'sensor_values.csv'
        self.controller_results = r'controller_results.csv'
        
        # Μεταβλητες Γραφων
        self.network_graph = None
        self.network_positions = None       
        self.sensor_graph = None
        
        # Μεταβλητες Αντικειμενων Δικτυου
        self.node_types = None
        self.manhole_nodes = None
        self.node_types = None
        self.node_types = None
        self.sensor_colors= []
        self.sensors_list = None
        
        self.num_of_sensors = None
        self.sensor_values = None
        
        # Μεταβλητες Απεικονησης Δικτυου
        self.fig = None
        self.ax = None        
        self.show_sensor_ids = True
        
        # Μεταβλητες αποκτησης αποτελεσματος
        self.mode_lookup = True
        self.lookup_tables = None
        self.lookup_table_path = r'LookUp Table.pickle'
        
        # Μεθοδοι αρχικοποιησης γραφου
        self.generate_sensor_objects()
        self.generate_pair_objects()
        self.create_graph()
        self.add_sensors()
        
    def generate_sensor_objects(self):
        # Αναγνωση Αρχειου Sensors
        csv_file_path = self.sensors_data_path
        df = pd.read_csv(csv_file_path, delimiter=',')
        
        # Δημιουργια Αντικειμενων Sensors
        sensors_list = []        
        i=0
        for index, csv_row in df.iterrows():
            new_sensor = Sensor(csv_row)  
            new_sensor.calculate_membership_params()
            new_sensor.define_input_membership_function()
            sensors_list.append(new_sensor)
            i+=1
            # if i>=56: break
        
        self.sensors_list = sensors_list
        print(f'Δημιουργηθηκαν {len(self.sensors_list)} αισθητηρες')
        
    def generate_pair_objects(self):
        # Αναγνωση του Πινακα Αναζητησης αν εχει επιλεψθει αυτη η λειτουργια
        if (self.mode_lookup is True):
            with open(self.lookup_table_path, 'rb') as handle:
                self.lookup_tables = pickle.load(handle)        
        
        # Αναγνωση pipe sensors αρχειου
        pairs_list = [] 
        csv_file_path = self.pipe_sensors_path
        df = pd.read_csv(csv_file_path, delimiter=',')  
        i=0      
        # Για καθε σωληνα δημιουργουντε τα ζευγαρια των αισθητηρων
        for pipe_index, csv_row in df.iterrows(): 
            sens_ids = csv_row['Sens_Ids']
            if (type(sens_ids) is float):continue # Αγνοηση Σωληνα χωρις αισθητηρες
            sens_ids = [int(x) for x in csv_row['Sens_Ids'].split('->')]
            sens_pairs = [(sens_ids[i], sens_ids[i + 1]) for i in range(len(sens_ids) - 1)] 
            # print(f'Pipe Index:{pipe_index} sens_ids:{sens_ids}')
            for pair_index,sens_pair in enumerate(sens_pairs):           
                first_sensor_id = sens_pair[0]
                second_sensor_id = sens_pair[1]
                            
                # print(f'      A_id:{first_sensor_id}  B_id:{second_sensor_id}')
                first_sensor = self.get_sensor_by_Id(self.sensors_list,first_sensor_id)
                second_sensor = self.get_sensor_by_Id(self.sensors_list,second_sensor_id)
                
                first_sensor.pipe_index = pipe_index
                second_sensor.pipe_index = pipe_index
            
                if (first_sensor is None):raise ValueError(f'No Sensor with id:{first_sensor_id}')
                if (second_sensor is None):raise ValueError(f'No Sensor with id:{second_sensor_id}')
            
                new_pair = Pair(first_sensor,second_sensor)
                new_pair.Pipe_Index = pipe_index
                new_pair.calculate_input_membership_params()
                new_pair.define_input_membership_function()
                new_pair.calculate_output_membership_params()
                new_pair.define_output_membership_function()
                if (self.mode_lookup is False):
                    new_pair.define_system_rules()
                else:
                    if (new_pair.Pair_Name in self.lookup_tables):
                        new_pair.result_matrix = self.lookup_tables[new_pair.Pair_Name]
                    else:
                        if (pair_index == 0 ):
                            raise ValueError(f'Pair 0 at pipe_index{pipe_index} has no lookup table in pickle file')
                        new_pair.result_matrix = pairs_list[-1].result_matrix
                
            
                pairs_list.append(new_pair)
            # if (i>=5):break
            
        self.pairs_list = pairs_list
        print(f'Δημιουργηθηκαν {len(self.pairs_list)} Ζευγαρια')        
        
               
    def create_graph(self):       
        # Αναγνωση αρχειου Δικτου
        csv_file_path = self.network_data_path
        df = pd.read_csv(csv_file_path, delimiter=',')
        
        # Δημιουργια Γραφου
        data = [(int(row[0]), int(row[1]), float(row[2])) for row in df.values]
        G = nx.DiGraph()
        i=0
        for edge in data:
            G.add_edge(edge[0], edge[1], weight=edge[2])
            i+=1
            # if (i>= 20):break

        network_positions = nx.kamada_kawai_layout(G, weight='weight',scale=0.50)
        nx.set_node_attributes(G, network_positions, 'pos')            
        self.network_graph = G
        self.network_positions = network_positions
        
        print('Δημιουργια Γραφου Δικτυου Ολοκληρωθηκε')

        
    # Μέθοδος Υπολογισμου του Αριθμου Αισθητηρων αναλογα με το μηκος του Σωληνα
    def calculate_flow_sensors(self,pipe_length, thresholds, sensor_counts):
        return next((count for thresh, count in zip(thresholds, sensor_counts) if pipe_length < thresh), sensor_counts[-1])              
    
    def add_sensors(self):
        graph = self.network_graph
        total_num_of_sensors = 0
        new_graph = nx.DiGraph()
        sensor_id = 1
        for u, v, data in graph.edges(data=True):
            length = data['weight']
            
            thresholds = [2.0, 10.0, 40.0, 64.0]    # Ορια μηκων σωληνων
            sensor_counts = [0, 2, 5, 6]            # Αριθμος Αισθητηρων για καθε ευρος
            number_of_sensors = self.calculate_flow_sensors(length, thresholds, sensor_counts)            
            total_num_of_sensors += number_of_sensors
            
            if (u not in new_graph):
                new_graph.add_node(u, pos=graph.nodes[u]['pos'],node_type='manhole')
            if (v not in new_graph):
                new_graph.add_node(v, pos=graph.nodes[v]['pos'],node_type='manhole')  
                
            middle_points = []
            start_pos = graph.nodes[u]['pos']
            end_pos = graph.nodes[v]['pos']            
           
            for i in range(1, number_of_sensors + 1):
                ratio = i / (number_of_sensors + 1.0)
                middle_point = (
                    (1 - ratio) * start_pos[0] + ratio * end_pos[0],
                    (1 - ratio) * start_pos[1] + ratio * end_pos[1]
                )
                
                new_graph.add_node(f'{sensor_id}', pos=middle_point,node_type='sensor')
                sensor_id += 1
                
                middle_points.append(middle_point)            
        
        self.sensor_graph = new_graph
        self.num_of_sensors = total_num_of_sensors
        self.sensors_positions = nx.get_node_attributes(self.sensor_graph, 'pos')
        
        self.node_types = nx.get_node_attributes(self.sensor_graph, 'node_type')
        self.manhole_nodes = [key for key, value in self.node_types.items() if value == 'manhole']
        self.sensor_nodes = [key for key, value in self.node_types.items() if value == 'sensor']
        self.sensor_colors = ['green' for _ in self.sensor_nodes]
        
        self.manhole_nodes_pos = nx.get_node_attributes(self.network_graph, 'pos')
        self.pipes_pos = nx.get_edge_attributes(self.network_graph, 'pos')
        
        print('Τοποθετηση Αισθητηρων σε Απεικονιση Ολοκληρωθηκε')
    
    def get_sensor_by_Id(self,sensor_list, Sensor_Id):
        for sensor in sensor_list:
            if (sensor.Sensor_Id == Sensor_Id):
                return sensor
        return None
    
    def execute_controller(self, event):    
        rand_index = r.randint(0, 10)
        self.sensor_colors[rand_index] = 'red'
        print(f'rand_index:{rand_index}')
        
        self.ax.cla()
        
        self.show_network()
        self.fig.canvas.draw_idle()  # Redraw the figure
             
    def toggle_sensor_name(self,event):
        self.show_sensor_ids = not(self.show_sensor_ids)  
        if (self.show_sensor_ids is True):
            self.button_toggle_sensor_name.label.set_text('Sensor Ids\nVisible')            
        else:
            self.button_toggle_sensor_name.label.set_text('Sensor Ids\nHidden')
        print(self.show_sensor_ids)
        
    def show_network(self):
        nx.draw_networkx_nodes(self.network_graph, self.network_positions,ax=self.ax,nodelist=self.manhole_nodes, node_size=50, node_color='skyblue')
        nx.draw_networkx_labels(self.network_graph, self.network_positions,ax=self.ax, labels={node: node for node in self.manhole_nodes}, font_size=8)
        nx.draw_networkx_edges(self.network_graph, self.network_positions,ax=self.ax, width=1.0, alpha=0.5, edge_color='gray')
        
        for sensor_node in zip(self.sensor_nodes):
            sensor_node = sensor_node[0]
            sensor_node_color = self.sensor_colors[int(sensor_node)]
            nx.draw_networkx_nodes(self.sensor_graph, self.sensors_positions,ax=self.ax, nodelist=[sensor_node], node_size=5, node_color=sensor_node_color)        
            if (self.show_sensor_ids is True):
                nx.draw_networkx_labels(self.sensor_graph, self.sensors_positions,ax=self.ax, labels={sensor_node: sensor_node}, font_size=7, font_color=sensor_node_color)

    def plot_network(self):
        # Create a figure and axis
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.2)  # Adjust bottom to make space for the button
        self.show_network()   

        
    def generate_random_sensor_values(self):
        sensor_values = [['Pipe_Index','Sensor_Id','Sensor_Value']]
        sensor_values_dict = {}
        for sensor in self.sensors_list:
            rand_val = random.uniform(sensor.QMin, sensor.QMax*0.999)            
            sensor_values_dict[sensor.Sensor_Id] = rand_val
            sensor_values.append([sensor.pipe_index,sensor.Sensor_Id,rand_val])
        self.sensor_values_dict = sensor_values_dict
        
        csv_file = self.sensor_values_path
        with open(csv_file, 'w', newline='') as file:
            writer = csv.writer(file)  
            for row in sensor_values:
                writer.writerow(row)       
                
    def read_sensor_values(self):
        csv_file_path = self.sensor_values_path
        df = pd.read_csv(csv_file_path, delimiter=',')  
        sensor_values = {}
        for index, csv_row in df.iterrows(): 
            _,Sensor_Id,Sensor_Value = csv_row['Pipe_Index'],csv_row['Sensor_Id'],csv_row['Sensor_Value']
            sensor_values[Sensor_Id] = Sensor_Value
        
        self.sensor_values_dict = sensor_values
        
    def result2color(self,world_result):
        if (world_result == 'LEAK_HIGH'):return [118/255.0,5/255.0,33/255.0]
        if (world_result == 'LEAK_MEDIUM'):return [218/255,104/255,83/255]
        if (world_result == 'LEAK_LOW'):return [250/255,202/255,177/255]
        if (world_result == 'ZERO'):return [0,0,0]
        if (world_result == 'INFLOW_LOW'):return [160/255,204/255,226/255]
        if (world_result == 'INFLOW_MEDIUM'):return [51/255,126/255,184/255]
        if (world_result == 'INFLOW_HIGH'):return [6/255,50/255,100/255]
        
    def exe_contr(self):
        detailed_resuts = [['Pair_Index','Results']]
        self.sensor_colors = {}
        
        res_count = {}
        for pair_index,pair in enumerate(self.pairs_list):
            s1_id = pair.Sensor_A.Sensor_Id
            s2_id = pair.Sensor_B.Sensor_Id
            s1_v = self.sensor_values_dict[s1_id]
            s2_v = self.sensor_values_dict[s2_id]
            s_diff = abs(s1_v-s2_v)
            res = pair.pprint_result(s1_v, s2_v, s_diff,self.mode_lookup)            
            self.sensor_colors[s1_id] = self.result2color(res[11])
            self.sensor_colors[s2_id] = self.result2color(res[11])
            
            detailed_resuts.append([pair_index,res])
            
            if (res[11] in res_count):
                res_count[res[11]] += 1
            else:
                res_count[res[11]]=0
                    
        return res_count
        
       
        
if __name__ == '__main__':
    
    main_system = sewage_system_main()    
    main_system.generate_random_sensor_values()
    main_system.read_sensor_values()
    result_compact = main_system.exe_contr()
    main_system.plot_network()
    print(result_compact)

        







