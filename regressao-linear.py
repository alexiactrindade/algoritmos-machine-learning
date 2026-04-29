from numpy import *

class RegressaoLinear:

#Intânciação dos atributos, através da palavra SELF! 
    def __init__(self, x, y):
        self.x = x
        self.y = y 
        self._correlation_coefficient = self._correlacao()
        self._inclination = self._inclinacao()
        self._intercept = self._interceptacao() 

#Cálculo da correlação que é a covariação de x,y dividida pela raiz da variança de x e y. 
    def _correlacao(self): 
        covariacao = cov(self.x, self.y, bias=True)[0][1]
        variancia_x = var(self.x)
        variancia_y = var(self.y)
        return covariacao / sqrt(variancia_x * variancia_y)

#Cálculo da formúla r(correlação)=(desvio padrão de y / desvio padrão de x)     
    def _inclinacao(self): 
        stdDeX = std(self.x)
        stdDeY = std(self.y)
        return self._correlation_coefficient * (stdDeY / stdDeX)
        
# ponto onde a reta cruza o eixo y 
    def _interceptacao(self): 
        mediaX = mean(self.x)
        mediaY = mean(self.y)
        return mediaY - mediaX * self._inclination 

#cálculo da previsão que corresponde a 
# Previsão = interceptação + (inclinação + valor que o usuário passa para o modelo)
    def previsao(self, valor):
        return self._intercept + (self._inclination * valor)


entrada_x = input("Digite os valores de X separados por vírgula: ")
entrada_y = input("Digite os valores de Y separados por vírgula: ")

lista_x = entrada_x.split(",")
lista_y = entrada_y.split(",")

x = array(lista_x, dtype=float)
y = array(lista_y, dtype=float)

modelo = RegressaoLinear(x, y)

valor = float(input("Digite um valor de X para prever Y: "))
print(modelo.previsao(valor)) 
