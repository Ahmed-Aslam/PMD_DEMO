import pandas as pd
from pyiron_base import PythonTemplateJob, DataContainer
import pandas
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import requests
from SPARQLWrapper import SPARQLWrapper, JSON
import os


class TensileJob(PythonTemplateJob):
    def __init__(self, project, job_name):
        super(TensileJob, self).__init__(project, job_name)
        self._experimental_json = None
        self._actual_data = None
        self._elast_min_ind = 0
        self._elast_max_ind = 0
        self._endpoint = None

    @property
    def experimental_json(self):
        return self._experimental_json

    @experimental_json.setter
    def experimental_json(self, data_frame):
        self._experimental_json = data_frame

    @property
    def actual_data_set(self):
        return self._actual_data

    @actual_data_set.setter
    def actual_data_set(self, data_frame):
        if isinstance(data_frame, pd.DataFrame):
            self._actual_data = data_frame
        else:
            raise TypeError("the dataset should be of type pandas.DataFrame")

    @property
    def endpoint(self):
        return self._endpoint

    @endpoint.setter
    def endpoint(self, url):
        self._endpoint = SPARQLWrapper(url)

    def query_data_source(self, test_name='tensile_test'):
        self.input.test_name = test_name
        query =f"""
        PREFIX pmdco: <https://material-digital.de/pmdco/>
        PREFIX tt: <https://material-digital.de/pmdao/tensile-test/>
        SELECT ?loc
        WHERE {{
        ?tt a tt:TensileTest .
        ?dr pmdco:hasDataResourceLocation ?loc .
        ?tt pmdco:hasDataResource ?dr .
        ?tt pmdco:hasSuppliedIdentifier ?s .
        }}"""
        self.endpoint.setQuery(query)
        self.endpoint.setReturnFormat(JSON)
        results = self.endpoint.query().convert()
        header2column = {}
        variables = results['head']['vars']
        for binding in results['results']['bindings']:
            for v in variables:
                if v not in header2column:
                    header2column[v] = []
                header2column[v] += [binding[v]['value']]
        _init_data_frame = pandas.DataFrame.from_dict(header2column)
        return _init_data_frame['loc'][0]

    def get_dataset(self, url):
        try:         
            content = requests.get(url).content.decode()
            self.experimental_json = pandas.read_json(content)
        except Exception as err_msg:
            raise Exception(f"Error: {err_msg}.Download of the file unsuccessful!,")

    def get_json(self):
        url = self._init_data_frame['link'][0]
        content = requests.get(url, headers={'PRIVATE-TOKEN': '{}'. format(os.environ['gitlab_token'])}).content
        self.experimental_json = pandas.read_json(content)

    def converter_strain(self, array):
        for ind,value in enumerate(array):
            try:
                array[ind] = float(value.replace(',', '.'))-0.0586 # this is to offset strain values to start from zero
            except:
                array[ind] = array[ind-1]
        return np.array(array, dtype='float32')
    
    def converter_stress(self, array):
        for ind,value in enumerate(array):
            array[ind] = float(value.replace(',', '.'))
        return np.array(array, dtype='float32')

    def extract_stress_strain(self):
        datalist = self._experimental_json['dataseries'][-1]['data']
        fields_units = self._experimental_json['dataseries'][-1]['fields']
        df_fields_units = pandas.DataFrame(fields_units, columns = ['fields', 'units']) 
        fields = list(df_fields_units['fields'])
        self._actual_data = pandas.DataFrame(datalist, columns = fields) 
        # values are started from iteration 2 to avoid negative strain
        # values are until 947 iteration as there is fracture after that
        self.input.strains = (self.converter_strain(np.array(self._actual_data['Extensometer elongation'][:])[2:947]))
        self.input.stresses = self.converter_stress(np.array(self._actual_data['Load'][:])[2:947])/120.6 # Area of specimen Zx1

    def get_linear_segment(self):
        strain_0 = 0.00
        #elastic_limit = 0.0009
        elastic_limit = 0.001
        self._elast_min_ind = 0
        self._elast_max_ind = 0
        flag_init = 0
        flag_end = 0
        i = 0
        while flag_init == 0 or flag_end == 0:
            if self.input.strains[i] >= strain_0 and flag_init == 0:
                self._elast_min_ind = i
                flag_init = 1
            if self.input.strains[i] > elastic_limit:
                self._elast_max_ind = i - 1
                flag_end = 1
            i = i + 1

    def plot_stress_strain(self):
        plt.xlabel('Displacement [%]')
        plt.ylabel('Stress [GPa]')
        plt.xlim(-0.01,1.9)
        plt.ylim(-0.0001,.45)
        plt.plot(self.input.strains, self.input.stresses, linewidth=4.0)
        plt.savefig('Stress_Strain.jpg', dpi=500)

    def calc_elastic_modulus(self):
        self.get_linear_segment()
        strains = self.input.strains * 0.01
        lm = LinearRegression()
        _x = strains[self._elast_min_ind:self._elast_max_ind]
        _y = self.input.stresses[self._elast_min_ind:self._elast_max_ind]
        _x = _x.reshape(-1, 1)
        _y = _y.reshape(-1, 1)
        lm.fit(_x, _y)
        self.output.elastic_modulus = float(lm.coef_[0])
    

    def run_static(self):
        self.calc_elastic_modulus()
        self.to_hdf()
        self.status.finished = True

    def update_triple_store(self, test_name="tensile_test"):
        # get quantity value of tensile test
        query = f"""
        PREFIX pmdco: <https://material-digital.de/pmdco/>
        PREFIX tt: <https://material-digital.de/pmdao/tensile-test/>
        prefix wikiba: <http://wikiba.se/ontology#>
        
        INSERT {{tt:hasMeasuredModulusOfElasticity  wikiba:quantityAmount  "{self.output.elastic_modulus}" }}
        
        
        WHERE {{
            ?tt pmdco:hasSuppliedIdentifier "{test_name}" .
        }}
        """
        self.endpoint.setQuery(query)
        self.endpoint.method = 'POST'
        self.endpoint.query()

    def verify_update(self):
        query = """
        PREFIX pmdco: <https://material-digital.de/pmdco/>
        PREFIX tt: <https://material-digital.de/pmdao/tensile-test/>
        prefix wikiba: <http://wikiba.se/ontology#>

        SELECT ?tt ?o
        WHERE{
           ?tt tt:hasMeasuredModulusOfElasticity ?o

        }
        """
        self.endpoint.setQuery(query)
        self.endpoint.method = 'GET'
        self.endpoint.setReturnFormat(JSON)
        results = self.endpoint.query().convert()
        #print(results)
        header2column = {}
        variables = results['head']['vars']
        for binding in results['results']['bindings']:
            for v in variables:
                if v not in header2column:
                    header2column[v] = []
                header2column[v] += [binding[v]['value']]

        df = pd.DataFrame.from_dict(header2column)
        #return(df)
        #if abs(float(df['E'].values[-1]) - self.output.elastic_modulus)/self.output.elastic_modulus < 0.001:
        #    print("correctly updated!")
        #    return True
        #else:
        #    print("the update was unsuccessful!")
        #    return False
       
