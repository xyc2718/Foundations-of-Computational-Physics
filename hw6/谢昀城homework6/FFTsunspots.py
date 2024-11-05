import numpy as np
import matplotlib.pyplot as plt

data = []

with open('sunspots.txt', 'r') as file:
    for line in file:
        # Split the line by whitespace and convert to a tuple of (index, value)
        index, value = line.split()
        data.append((int(index), float(value)))

data=np.array(data)

x=data[:,0]
y=data[:,1]
x=x/12 #将月份转换为年份

# 执行 FFT
y_fft = np.fft.fft(y)
# 获取频率
frequencies = np.fft.fftfreq(len(y), d=1/12) 
bigen=len(y_fft)//2+1
y_fft=y_fft[:bigen]
y_fft=y_fft/sum(abs(y_fft))
frequencies=frequencies[:bigen]
y_fft[0]=0 #去掉直流分量
max_freq = frequencies[np.argmax(np.abs(y_fft))]

# 绘制结果
fig=plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(frequencies, np.abs(y_fft))
plt.title('FFT of the Sunspots Signal')
plt.xlabel('Frequency')
plt.ylabel('Magnitude')
plt.xlim(0, np.max(frequencies)/2)  # 只显示正频率
plt.scatter(max_freq, np.abs(y_fft[np.argmax(np.abs(y_fft))]), color='red')
plt.legend(['FFT of the Sunspots Signal', 'Peak Frequency'])

plt.subplot(1, 2, 2)
plt.plot(frequencies, np.abs(y_fft))
plt.title('FFT of the Sunspots Signal Near the Peak')
plt.scatter(max_freq, np.abs(y_fft[np.argmax(np.abs(y_fft))]), color='red')
plt.legend(['FFT of the Sunspots Signal', 'Peak Frequency'])
plt.xlim(0, 0.5)
fig.savefig('FFT of sunspots.png',dpi=1000)


fig=plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(x,y)
plt.title('Sunspot Signal')
plt.xlabel('Time')
plt.ylabel('Value')
plt.subplot(1, 2, 2)
re=y_fft[np.argmax(np.abs(y_fft))]
a=np.real(re)
b=np.imag(re)
ym=a*np.cos(2*np.pi*max_freq*x)+b*np.sin(2*np.pi*max_freq*x)
plt.plot(x,ym,linewidth=1)
plt.title("Dominant Frequency Component")
fig.savefig('Original Signal and Dominant Component.png',dpi=1000)

print(f'The dominant frequency is {max_freq:.5f} per year')
print(f'The dominant period is {1/max_freq:.2f} years')