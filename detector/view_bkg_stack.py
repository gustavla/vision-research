
import numpy as np
import gv
import amitgroup as ag
import matplotlib.pylab as plt
import sys

parts_file = gv.parts_descriptor.PartsDescriptor.load(sys.argv[1])
parts = parts_file.parts

print 'parts', parts.shape

data = np.load(sys.argv[2])
index = int(sys.argv[3])
bkg_stack, bkg_stack_num = data['bkg_stack'], data['bkg_stack_num']

part = parts[index-1,...,:4] + parts[index-1,...,4:]

stack = bkg_stack[index,:bkg_stack_num[index]]

mean_stack = stack.mean(axis=0)

plt.figure(figsize=(13,8))
ax = plt.subplot2grid((7,10), (0, 0), colspan=4, rowspan=4).set_axis_off()
plt.imshow(mean_stack, interpolation='nearest', cmap=plt.cm.gray)
plt.title('Mean part')

for i in xrange(4):
    plt.subplot2grid((7,10), (2* (i//2), 6+(i%2)*2), colspan=2, rowspan=2).set_axis_off()
    plt.imshow(part[...,i], interpolation='nearest', vmin=0, vmax=1)
    plt.title(['Horizontal probs.', 'Diag. probs.', 'Vertical probs.', 'Diag. probs'][i])
    #if i == 3:
    plt.colorbar()


for i in xrange(len(stack)):
    plt.subplot2grid((7,10), (5+i//10, i%10)).set_axis_off()
    plt.imshow(stack[i], interpolation='nearest', cmap=plt.cm.gray)
    if i == 4:
        plt.title('Original background parts')

#plt.show()
plt.savefig('part-number-{0}.png'.format(index))
plt.savefig('part-number-{0}.pdf'.format(index))
