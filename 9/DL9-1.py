class MulLayer:
    def __init__(self):
        self.x=None
        self.y=None
    def forward(self,x,y):
        self.x=x
        self.y=y
        out=x*y
        return out

    def backward(self,dout): #dout에 1전달
        dx=dout*self.y
        dy=dout*self.x
        return dx,dy #dx<-dapple_price, dy<-dtax

apple=100
apple_num=2
tax=1.1

mul_apple_layer=MulLayer()
mul_tax_layer=MulLayer()

#forward
apple_price = mul_apple_layer.forward(apple,apple_num)
price=mul_tax_layer.forward(apple_price,tax)

#backward
dprice=1
dapple_price,dtax= mul_tax_layer.backward(dprice)
dapple,dapple_num = mul_apple_layer.backward(dapple_price)


print("price:",int(price))
print("dApple:",dapple)
print("dApple_num:",int(dapple_num))
print("dTax:",int(dtax))