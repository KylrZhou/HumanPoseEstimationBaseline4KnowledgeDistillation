from torch.nn import Module

class ModelMergeBNH(Module):
    def __init__(self, Backbone, Neck, Head):
        super(ModelMergeBNH, self).__init__()
        self.Backbone = Backbone
        self.Neck = Neck
        self.Head = Head
        
    def forward(self, x):
        x = self.Backbone(x)
        x = self.Neck(x)
        x = self.Head(x)
        return x

class ModelMergeBH(Module):
    def __init__(self, Backbone, Head):
        super(ModelMergeBH, self).__init__()
        self.Backbone = Backbone
        self.Head = Head
        
    def forward(self, x):
        x = self.Backbone(x)
        x = self.Head(x)
        return x
    
def ModelMerge(Backbone, Neck, Head, bypass = None):
    """
    Backbone, Neck, Head: type(nn.Module or None) model parts you want to merge
    bypass: type(str) value('Backbone', 'Neck', 'Head') bypass other parts and return selected part only
        |bypass will automatically enable if only Backbone is not None
    """
    if bypass != None:
        return eval(f'{bypass}')
    elif Neck != None:
        return MergeBNH(Backbone, Neck, Head)
    elif Head != None:
        return MergeBH(Backbone, Head)
    return Backbone