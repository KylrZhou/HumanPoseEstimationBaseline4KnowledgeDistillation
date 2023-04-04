from json import dump

class loggr():
    def __init__(self, Write2Path = None):
        self.Write2Path = Write2Path
        self.data = {}
        
    def update(key, data):
        self.data[key] = data
            
    def print2terminal(strs):
        text = ''
        for k, v in strs.items():
            text += f'{k} : {v} '
        print(strs)
    
    def wrt():
        if self.Write2Path is not None:
            strs = {}
            strs['Mode'] = self.data['MODE']
            strs['Epoch'] = self.data['EPOCH']
            strs['Data_time'] = self.data['DATATIME']
            strs['Batch_time'] = self.data['BATCHTIME']
            for k, v in self.data['LOSS'].items():
                strs[k] = v
            for k, v in self.data['ACC'].items():
                strs[k] = v
            with open(self.Write2Path, 'a') as f:
                dump(strs,f)
        self.print2terminal(strs)
        self.data = {}