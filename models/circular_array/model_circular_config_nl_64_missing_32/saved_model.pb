??
??
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?"serve*2.4.12v2.4.0-49-g85c8b2a817f8??

conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d/kernel
x
!conv2d/kernel/Read/ReadVariableOpReadVariableOpconv2d/kernel*'
_output_shapes
:?*
dtype0
o
conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d/bias
h
conv2d/bias/Read/ReadVariableOpReadVariableOpconv2d/bias*
_output_shapes	
:?*
dtype0
{
p_re_lu/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_namep_re_lu/alpha
t
!p_re_lu/alpha/Read/ReadVariableOpReadVariableOpp_re_lu/alpha*#
_output_shapes
:?*
dtype0
?
conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??* 
shared_nameconv2d_1/kernel
}
#conv2d_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_1/kernel*(
_output_shapes
:??*
dtype0
s
conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_1/bias
l
!conv2d_1/bias/Read/ReadVariableOpReadVariableOpconv2d_1/bias*
_output_shapes	
:?*
dtype0

p_re_lu_1/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namep_re_lu_1/alpha
x
#p_re_lu_1/alpha/Read/ReadVariableOpReadVariableOpp_re_lu_1/alpha*#
_output_shapes
:?*
dtype0
?
conv2d_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??* 
shared_nameconv2d_2/kernel
}
#conv2d_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_2/kernel*(
_output_shapes
:??*
dtype0
s
conv2d_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*
shared_nameconv2d_2/bias
l
!conv2d_2/bias/Read/ReadVariableOpReadVariableOpconv2d_2/bias*
_output_shapes	
:?*
dtype0

p_re_lu_2/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namep_re_lu_2/alpha
x
#p_re_lu_2/alpha/Read/ReadVariableOpReadVariableOpp_re_lu_2/alpha*#
_output_shapes
:?*
dtype0
?
conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??*(
shared_nameconv2d_transpose/kernel
?
+conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose/kernel*(
_output_shapes
:??*
dtype0
?
conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*&
shared_nameconv2d_transpose/bias
|
)conv2d_transpose/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose/bias*
_output_shapes	
:?*
dtype0

p_re_lu_3/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namep_re_lu_3/alpha
x
#p_re_lu_3/alpha/Read/ReadVariableOpReadVariableOpp_re_lu_3/alpha*#
_output_shapes
:?*
dtype0
?
conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??**
shared_nameconv2d_transpose_1/kernel
?
-conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/kernel*(
_output_shapes
:??*
dtype0
?
conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameconv2d_transpose_1/bias
?
+conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_1/bias*
_output_shapes	
:?*
dtype0

p_re_lu_4/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:?* 
shared_namep_re_lu_4/alpha
x
#p_re_lu_4/alpha/Read/ReadVariableOpReadVariableOpp_re_lu_4/alpha*#
_output_shapes
:?*
dtype0
?
conv2d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:??**
shared_nameconv2d_transpose_2/kernel
?
-conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/kernel*(
_output_shapes
:??*
dtype0
?
conv2d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*(
shared_nameconv2d_transpose_2/bias
?
+conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_2/bias*
_output_shapes	
:?*
dtype0

p_re_lu_5/alphaVarHandleOp*
_output_shapes
: *
dtype0*
shape:@??* 
shared_namep_re_lu_5/alpha
x
#p_re_lu_5/alpha/Read/ReadVariableOpReadVariableOpp_re_lu_5/alpha*#
_output_shapes
:@??*
dtype0
?
conv2d_transpose_3/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:?**
shared_nameconv2d_transpose_3/kernel
?
-conv2d_transpose_3/kernel/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/kernel*'
_output_shapes
:?*
dtype0
?
conv2d_transpose_3/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*(
shared_nameconv2d_transpose_3/bias

+conv2d_transpose_3/bias/Read/ReadVariableOpReadVariableOpconv2d_transpose_3/bias*
_output_shapes
:*
dtype0

NoOpNoOp
?;
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?;
value?;B?; B?;
?
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer-9
layer_with_weights-8
layer-10
layer_with_weights-9
layer-11
layer_with_weights-10
layer-12
layer_with_weights-11
layer-13
layer_with_weights-12
layer-14
trainable_variables
	variables
regularization_losses
	keras_api

signatures
 
h

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
]
	alpha
regularization_losses
trainable_variables
	variables
	keras_api
h

 kernel
!bias
"regularization_losses
#trainable_variables
$	variables
%	keras_api
]
	&alpha
'regularization_losses
(trainable_variables
)	variables
*	keras_api
h

+kernel
,bias
-regularization_losses
.trainable_variables
/	variables
0	keras_api
]
	1alpha
2regularization_losses
3trainable_variables
4	variables
5	keras_api
h

6kernel
7bias
8regularization_losses
9trainable_variables
:	variables
;	keras_api
]
	<alpha
=regularization_losses
>trainable_variables
?	variables
@	keras_api
R
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
h

Ekernel
Fbias
Gregularization_losses
Htrainable_variables
I	variables
J	keras_api
]
	Kalpha
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
h

Pkernel
Qbias
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
]
	Valpha
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
h

[kernel
\bias
]regularization_losses
^trainable_variables
_	variables
`	keras_api
?
0
1
2
 3
!4
&5
+6
,7
18
69
710
<11
E12
F13
K14
P15
Q16
V17
[18
\19
?
0
1
2
 3
!4
&5
+6
,7
18
69
710
<11
E12
F13
K14
P15
Q16
V17
[18
\19
 
?

alayers
bnon_trainable_variables
clayer_regularization_losses
dmetrics
elayer_metrics
trainable_variables
	variables
regularization_losses
 
YW
VARIABLE_VALUEconv2d/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
US
VARIABLE_VALUEconv2d/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE
 

0
1

0
1
?
fnon_trainable_variables
regularization_losses
glayer_regularization_losses
hlayer_metrics
imetrics
trainable_variables
	variables

jlayers
XV
VARIABLE_VALUEp_re_lu/alpha5layer_with_weights-1/alpha/.ATTRIBUTES/VARIABLE_VALUE
 

0

0
?
knon_trainable_variables
regularization_losses
llayer_regularization_losses
mlayer_metrics
nmetrics
trainable_variables
	variables

olayers
[Y
VARIABLE_VALUEconv2d_1/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_1/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE
 

 0
!1

 0
!1
?
pnon_trainable_variables
"regularization_losses
qlayer_regularization_losses
rlayer_metrics
smetrics
#trainable_variables
$	variables

tlayers
ZX
VARIABLE_VALUEp_re_lu_1/alpha5layer_with_weights-3/alpha/.ATTRIBUTES/VARIABLE_VALUE
 

&0

&0
?
unon_trainable_variables
'regularization_losses
vlayer_regularization_losses
wlayer_metrics
xmetrics
(trainable_variables
)	variables

ylayers
[Y
VARIABLE_VALUEconv2d_2/kernel6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUE
WU
VARIABLE_VALUEconv2d_2/bias4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUE
 

+0
,1

+0
,1
?
znon_trainable_variables
-regularization_losses
{layer_regularization_losses
|layer_metrics
}metrics
.trainable_variables
/	variables

~layers
ZX
VARIABLE_VALUEp_re_lu_2/alpha5layer_with_weights-5/alpha/.ATTRIBUTES/VARIABLE_VALUE
 

10

10
?
non_trainable_variables
2regularization_losses
 ?layer_regularization_losses
?layer_metrics
?metrics
3trainable_variables
4	variables
?layers
ca
VARIABLE_VALUEconv2d_transpose/kernel6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUE
_]
VARIABLE_VALUEconv2d_transpose/bias4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUE
 

60
71

60
71
?
?non_trainable_variables
8regularization_losses
 ?layer_regularization_losses
?layer_metrics
?metrics
9trainable_variables
:	variables
?layers
ZX
VARIABLE_VALUEp_re_lu_3/alpha5layer_with_weights-7/alpha/.ATTRIBUTES/VARIABLE_VALUE
 

<0

<0
?
?non_trainable_variables
=regularization_losses
 ?layer_regularization_losses
?layer_metrics
?metrics
>trainable_variables
?	variables
?layers
 
 
 
?
?non_trainable_variables
Aregularization_losses
 ?layer_regularization_losses
?layer_metrics
?metrics
Btrainable_variables
C	variables
?layers
ec
VARIABLE_VALUEconv2d_transpose_1/kernel6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUE
a_
VARIABLE_VALUEconv2d_transpose_1/bias4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUE
 

E0
F1

E0
F1
?
?non_trainable_variables
Gregularization_losses
 ?layer_regularization_losses
?layer_metrics
?metrics
Htrainable_variables
I	variables
?layers
ZX
VARIABLE_VALUEp_re_lu_4/alpha5layer_with_weights-9/alpha/.ATTRIBUTES/VARIABLE_VALUE
 

K0

K0
?
?non_trainable_variables
Lregularization_losses
 ?layer_regularization_losses
?layer_metrics
?metrics
Mtrainable_variables
N	variables
?layers
fd
VARIABLE_VALUEconv2d_transpose_2/kernel7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_2/bias5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUE
 

P0
Q1

P0
Q1
?
?non_trainable_variables
Rregularization_losses
 ?layer_regularization_losses
?layer_metrics
?metrics
Strainable_variables
T	variables
?layers
[Y
VARIABLE_VALUEp_re_lu_5/alpha6layer_with_weights-11/alpha/.ATTRIBUTES/VARIABLE_VALUE
 

V0

V0
?
?non_trainable_variables
Wregularization_losses
 ?layer_regularization_losses
?layer_metrics
?metrics
Xtrainable_variables
Y	variables
?layers
fd
VARIABLE_VALUEconv2d_transpose_3/kernel7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUE
b`
VARIABLE_VALUEconv2d_transpose_3/bias5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUE
 

[0
\1

[0
\1
?
?non_trainable_variables
]regularization_losses
 ?layer_regularization_losses
?layer_metrics
?metrics
^trainable_variables
_	variables
?layers
n
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????@?*
dtype0*$
shape:?????????@?
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1conv2d/kernelconv2d/biasp_re_lu/alphaconv2d_1/kernelconv2d_1/biasp_re_lu_1/alphaconv2d_2/kernelconv2d_2/biasp_re_lu_2/alphaconv2d_transpose/kernelconv2d_transpose/biasp_re_lu_3/alphaconv2d_transpose_1/kernelconv2d_transpose_1/biasp_re_lu_4/alphaconv2d_transpose_2/kernelconv2d_transpose_2/biasp_re_lu_5/alphaconv2d_transpose_3/kernelconv2d_transpose_3/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@?*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? */
f*R(
&__inference_signature_wrapper_45385508
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename!conv2d/kernel/Read/ReadVariableOpconv2d/bias/Read/ReadVariableOp!p_re_lu/alpha/Read/ReadVariableOp#conv2d_1/kernel/Read/ReadVariableOp!conv2d_1/bias/Read/ReadVariableOp#p_re_lu_1/alpha/Read/ReadVariableOp#conv2d_2/kernel/Read/ReadVariableOp!conv2d_2/bias/Read/ReadVariableOp#p_re_lu_2/alpha/Read/ReadVariableOp+conv2d_transpose/kernel/Read/ReadVariableOp)conv2d_transpose/bias/Read/ReadVariableOp#p_re_lu_3/alpha/Read/ReadVariableOp-conv2d_transpose_1/kernel/Read/ReadVariableOp+conv2d_transpose_1/bias/Read/ReadVariableOp#p_re_lu_4/alpha/Read/ReadVariableOp-conv2d_transpose_2/kernel/Read/ReadVariableOp+conv2d_transpose_2/bias/Read/ReadVariableOp#p_re_lu_5/alpha/Read/ReadVariableOp-conv2d_transpose_3/kernel/Read/ReadVariableOp+conv2d_transpose_3/bias/Read/ReadVariableOpConst*!
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_save_45386044
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameconv2d/kernelconv2d/biasp_re_lu/alphaconv2d_1/kernelconv2d_1/biasp_re_lu_1/alphaconv2d_2/kernelconv2d_2/biasp_re_lu_2/alphaconv2d_transpose/kernelconv2d_transpose/biasp_re_lu_3/alphaconv2d_transpose_1/kernelconv2d_transpose_1/biasp_re_lu_4/alphaconv2d_transpose_2/kernelconv2d_transpose_2/biasp_re_lu_5/alphaconv2d_transpose_3/kernelconv2d_transpose_3/bias* 
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *-
f(R&
$__inference__traced_restore_45386114??
?U
?

$__inference__traced_restore_45386114
file_prefix"
assignvariableop_conv2d_kernel"
assignvariableop_1_conv2d_bias$
 assignvariableop_2_p_re_lu_alpha&
"assignvariableop_3_conv2d_1_kernel$
 assignvariableop_4_conv2d_1_bias&
"assignvariableop_5_p_re_lu_1_alpha&
"assignvariableop_6_conv2d_2_kernel$
 assignvariableop_7_conv2d_2_bias&
"assignvariableop_8_p_re_lu_2_alpha.
*assignvariableop_9_conv2d_transpose_kernel-
)assignvariableop_10_conv2d_transpose_bias'
#assignvariableop_11_p_re_lu_3_alpha1
-assignvariableop_12_conv2d_transpose_1_kernel/
+assignvariableop_13_conv2d_transpose_1_bias'
#assignvariableop_14_p_re_lu_4_alpha1
-assignvariableop_15_conv2d_transpose_2_kernel/
+assignvariableop_16_conv2d_transpose_2_bias'
#assignvariableop_17_p_re_lu_5_alpha1
-assignvariableop_18_conv2d_transpose_3_kernel/
+assignvariableop_19_conv2d_transpose_3_bias
identity_21??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_3?AssignVariableOp_4?AssignVariableOp_5?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?	
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?	
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/alpha/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/alpha/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/alpha/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/alpha/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/alpha/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/alpha/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*h
_output_shapesV
T:::::::::::::::::::::*#
dtypes
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_conv2d_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_conv2d_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOp assignvariableop_2_p_re_lu_alphaIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOp"assignvariableop_3_conv2d_1_kernelIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp assignvariableop_4_conv2d_1_biasIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp"assignvariableop_5_p_re_lu_1_alphaIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp"assignvariableop_6_conv2d_2_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp assignvariableop_7_conv2d_2_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp"assignvariableop_8_p_re_lu_2_alphaIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp*assignvariableop_9_conv2d_transpose_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp)assignvariableop_10_conv2d_transpose_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp#assignvariableop_11_p_re_lu_3_alphaIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp-assignvariableop_12_conv2d_transpose_1_kernelIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp+assignvariableop_13_conv2d_transpose_1_biasIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp#assignvariableop_14_p_re_lu_4_alphaIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOp-assignvariableop_15_conv2d_transpose_2_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOp+assignvariableop_16_conv2d_transpose_2_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOp#assignvariableop_17_p_re_lu_5_alphaIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOp-assignvariableop_18_conv2d_transpose_3_kernelIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOp+assignvariableop_19_conv2d_transpose_3_biasIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_199
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?
Identity_20Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_20?
Identity_21IdentityIdentity_20:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_21"#
identity_21Identity_21:output:0*e
_input_shapesT
R: ::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
??
?
#__inference__wrapped_model_45384746
input_1/
+model_conv2d_conv2d_readvariableop_resource0
,model_conv2d_biasadd_readvariableop_resource)
%model_p_re_lu_readvariableop_resource1
-model_conv2d_1_conv2d_readvariableop_resource2
.model_conv2d_1_biasadd_readvariableop_resource+
'model_p_re_lu_1_readvariableop_resource1
-model_conv2d_2_conv2d_readvariableop_resource2
.model_conv2d_2_biasadd_readvariableop_resource+
'model_p_re_lu_2_readvariableop_resourceC
?model_conv2d_transpose_conv2d_transpose_readvariableop_resource:
6model_conv2d_transpose_biasadd_readvariableop_resource+
'model_p_re_lu_3_readvariableop_resourceE
Amodel_conv2d_transpose_1_conv2d_transpose_readvariableop_resource<
8model_conv2d_transpose_1_biasadd_readvariableop_resource+
'model_p_re_lu_4_readvariableop_resourceE
Amodel_conv2d_transpose_2_conv2d_transpose_readvariableop_resource<
8model_conv2d_transpose_2_biasadd_readvariableop_resource+
'model_p_re_lu_5_readvariableop_resourceE
Amodel_conv2d_transpose_3_conv2d_transpose_readvariableop_resource<
8model_conv2d_transpose_3_biasadd_readvariableop_resource
identity??#model/conv2d/BiasAdd/ReadVariableOp?"model/conv2d/Conv2D/ReadVariableOp?%model/conv2d_1/BiasAdd/ReadVariableOp?$model/conv2d_1/Conv2D/ReadVariableOp?%model/conv2d_2/BiasAdd/ReadVariableOp?$model/conv2d_2/Conv2D/ReadVariableOp?-model/conv2d_transpose/BiasAdd/ReadVariableOp?6model/conv2d_transpose/conv2d_transpose/ReadVariableOp?/model/conv2d_transpose_1/BiasAdd/ReadVariableOp?8model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?/model/conv2d_transpose_2/BiasAdd/ReadVariableOp?8model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?/model/conv2d_transpose_3/BiasAdd/ReadVariableOp?8model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?model/p_re_lu/ReadVariableOp?model/p_re_lu_1/ReadVariableOp?model/p_re_lu_2/ReadVariableOp?model/p_re_lu_3/ReadVariableOp?model/p_re_lu_4/ReadVariableOp?model/p_re_lu_5/ReadVariableOp?
"model/conv2d/Conv2D/ReadVariableOpReadVariableOp+model_conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02$
"model/conv2d/Conv2D/ReadVariableOp?
model/conv2d/Conv2DConv2Dinput_1*model/conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
model/conv2d/Conv2D?
#model/conv2d/BiasAdd/ReadVariableOpReadVariableOp,model_conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02%
#model/conv2d/BiasAdd/ReadVariableOp?
model/conv2d/BiasAddBiasAddmodel/conv2d/Conv2D:output:0+model/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model/conv2d/BiasAdd?
model/p_re_lu/ReluRelumodel/conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/p_re_lu/Relu?
model/p_re_lu/ReadVariableOpReadVariableOp%model_p_re_lu_readvariableop_resource*#
_output_shapes
:?*
dtype02
model/p_re_lu/ReadVariableOp?
model/p_re_lu/NegNeg$model/p_re_lu/ReadVariableOp:value:0*
T0*#
_output_shapes
:?2
model/p_re_lu/Neg?
model/p_re_lu/Neg_1Negmodel/conv2d/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/p_re_lu/Neg_1?
model/p_re_lu/Relu_1Relumodel/p_re_lu/Neg_1:y:0*
T0*0
_output_shapes
:??????????2
model/p_re_lu/Relu_1?
model/p_re_lu/mulMulmodel/p_re_lu/Neg:y:0"model/p_re_lu/Relu_1:activations:0*
T0*0
_output_shapes
:??????????2
model/p_re_lu/mul?
model/p_re_lu/addAddV2 model/p_re_lu/Relu:activations:0model/p_re_lu/mul:z:0*
T0*0
_output_shapes
:??????????2
model/p_re_lu/add?
$model/conv2d_1/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02&
$model/conv2d_1/Conv2D/ReadVariableOp?
model/conv2d_1/Conv2DConv2Dmodel/p_re_lu/add:z:0,model/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
model/conv2d_1/Conv2D?
%model/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%model/conv2d_1/BiasAdd/ReadVariableOp?
model/conv2d_1/BiasAddBiasAddmodel/conv2d_1/Conv2D:output:0-model/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model/conv2d_1/BiasAdd?
model/p_re_lu_1/ReluRelumodel/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/p_re_lu_1/Relu?
model/p_re_lu_1/ReadVariableOpReadVariableOp'model_p_re_lu_1_readvariableop_resource*#
_output_shapes
:?*
dtype02 
model/p_re_lu_1/ReadVariableOp?
model/p_re_lu_1/NegNeg&model/p_re_lu_1/ReadVariableOp:value:0*
T0*#
_output_shapes
:?2
model/p_re_lu_1/Neg?
model/p_re_lu_1/Neg_1Negmodel/conv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/p_re_lu_1/Neg_1?
model/p_re_lu_1/Relu_1Relumodel/p_re_lu_1/Neg_1:y:0*
T0*0
_output_shapes
:??????????2
model/p_re_lu_1/Relu_1?
model/p_re_lu_1/mulMulmodel/p_re_lu_1/Neg:y:0$model/p_re_lu_1/Relu_1:activations:0*
T0*0
_output_shapes
:??????????2
model/p_re_lu_1/mul?
model/p_re_lu_1/addAddV2"model/p_re_lu_1/Relu:activations:0model/p_re_lu_1/mul:z:0*
T0*0
_output_shapes
:??????????2
model/p_re_lu_1/add?
$model/conv2d_2/Conv2D/ReadVariableOpReadVariableOp-model_conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02&
$model/conv2d_2/Conv2D/ReadVariableOp?
model/conv2d_2/Conv2DConv2Dmodel/p_re_lu_1/add:z:0,model/conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
model/conv2d_2/Conv2D?
%model/conv2d_2/BiasAdd/ReadVariableOpReadVariableOp.model_conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02'
%model/conv2d_2/BiasAdd/ReadVariableOp?
model/conv2d_2/BiasAddBiasAddmodel/conv2d_2/Conv2D:output:0-model/conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
model/conv2d_2/BiasAdd?
model/p_re_lu_2/ReluRelumodel/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/p_re_lu_2/Relu?
model/p_re_lu_2/ReadVariableOpReadVariableOp'model_p_re_lu_2_readvariableop_resource*#
_output_shapes
:?*
dtype02 
model/p_re_lu_2/ReadVariableOp?
model/p_re_lu_2/NegNeg&model/p_re_lu_2/ReadVariableOp:value:0*
T0*#
_output_shapes
:?2
model/p_re_lu_2/Neg?
model/p_re_lu_2/Neg_1Negmodel/conv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/p_re_lu_2/Neg_1?
model/p_re_lu_2/Relu_1Relumodel/p_re_lu_2/Neg_1:y:0*
T0*0
_output_shapes
:??????????2
model/p_re_lu_2/Relu_1?
model/p_re_lu_2/mulMulmodel/p_re_lu_2/Neg:y:0$model/p_re_lu_2/Relu_1:activations:0*
T0*0
_output_shapes
:??????????2
model/p_re_lu_2/mul?
model/p_re_lu_2/addAddV2"model/p_re_lu_2/Relu:activations:0model/p_re_lu_2/mul:z:0*
T0*0
_output_shapes
:??????????2
model/p_re_lu_2/add?
model/conv2d_transpose/ShapeShapemodel/p_re_lu_2/add:z:0*
T0*
_output_shapes
:2
model/conv2d_transpose/Shape?
*model/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2,
*model/conv2d_transpose/strided_slice/stack?
,model/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/conv2d_transpose/strided_slice/stack_1?
,model/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2.
,model/conv2d_transpose/strided_slice/stack_2?
$model/conv2d_transpose/strided_sliceStridedSlice%model/conv2d_transpose/Shape:output:03model/conv2d_transpose/strided_slice/stack:output:05model/conv2d_transpose/strided_slice/stack_1:output:05model/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2&
$model/conv2d_transpose/strided_slice?
model/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2 
model/conv2d_transpose/stack/1?
model/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2 
model/conv2d_transpose/stack/2?
model/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2 
model/conv2d_transpose/stack/3?
model/conv2d_transpose/stackPack-model/conv2d_transpose/strided_slice:output:0'model/conv2d_transpose/stack/1:output:0'model/conv2d_transpose/stack/2:output:0'model/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
model/conv2d_transpose/stack?
,model/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,model/conv2d_transpose/strided_slice_1/stack?
.model/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv2d_transpose/strided_slice_1/stack_1?
.model/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv2d_transpose/strided_slice_1/stack_2?
&model/conv2d_transpose/strided_slice_1StridedSlice%model/conv2d_transpose/stack:output:05model/conv2d_transpose/strided_slice_1/stack:output:07model/conv2d_transpose/strided_slice_1/stack_1:output:07model/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&model/conv2d_transpose/strided_slice_1?
6model/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp?model_conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype028
6model/conv2d_transpose/conv2d_transpose/ReadVariableOp?
'model/conv2d_transpose/conv2d_transposeConv2DBackpropInput%model/conv2d_transpose/stack:output:0>model/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0model/p_re_lu_2/add:z:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2)
'model/conv2d_transpose/conv2d_transpose?
-model/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp6model_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02/
-model/conv2d_transpose/BiasAdd/ReadVariableOp?
model/conv2d_transpose/BiasAddBiasAdd0model/conv2d_transpose/conv2d_transpose:output:05model/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2 
model/conv2d_transpose/BiasAdd?
model/p_re_lu_3/ReluRelu'model/conv2d_transpose/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/p_re_lu_3/Relu?
model/p_re_lu_3/ReadVariableOpReadVariableOp'model_p_re_lu_3_readvariableop_resource*#
_output_shapes
:?*
dtype02 
model/p_re_lu_3/ReadVariableOp?
model/p_re_lu_3/NegNeg&model/p_re_lu_3/ReadVariableOp:value:0*
T0*#
_output_shapes
:?2
model/p_re_lu_3/Neg?
model/p_re_lu_3/Neg_1Neg'model/conv2d_transpose/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/p_re_lu_3/Neg_1?
model/p_re_lu_3/Relu_1Relumodel/p_re_lu_3/Neg_1:y:0*
T0*0
_output_shapes
:??????????2
model/p_re_lu_3/Relu_1?
model/p_re_lu_3/mulMulmodel/p_re_lu_3/Neg:y:0$model/p_re_lu_3/Relu_1:activations:0*
T0*0
_output_shapes
:??????????2
model/p_re_lu_3/mul?
model/p_re_lu_3/addAddV2"model/p_re_lu_3/Relu:activations:0model/p_re_lu_3/mul:z:0*
T0*0
_output_shapes
:??????????2
model/p_re_lu_3/add?
model/add/addAddV2model/p_re_lu_3/add:z:0model/p_re_lu_1/add:z:0*
T0*0
_output_shapes
:??????????2
model/add/add?
model/conv2d_transpose_1/ShapeShapemodel/add/add:z:0*
T0*
_output_shapes
:2 
model/conv2d_transpose_1/Shape?
,model/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,model/conv2d_transpose_1/strided_slice/stack?
.model/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv2d_transpose_1/strided_slice/stack_1?
.model/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv2d_transpose_1/strided_slice/stack_2?
&model/conv2d_transpose_1/strided_sliceStridedSlice'model/conv2d_transpose_1/Shape:output:05model/conv2d_transpose_1/strided_slice/stack:output:07model/conv2d_transpose_1/strided_slice/stack_1:output:07model/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&model/conv2d_transpose_1/strided_slice?
 model/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 model/conv2d_transpose_1/stack/1?
 model/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2"
 model/conv2d_transpose_1/stack/2?
 model/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2"
 model/conv2d_transpose_1/stack/3?
model/conv2d_transpose_1/stackPack/model/conv2d_transpose_1/strided_slice:output:0)model/conv2d_transpose_1/stack/1:output:0)model/conv2d_transpose_1/stack/2:output:0)model/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2 
model/conv2d_transpose_1/stack?
.model/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model/conv2d_transpose_1/strided_slice_1/stack?
0model/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0model/conv2d_transpose_1/strided_slice_1/stack_1?
0model/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model/conv2d_transpose_1/strided_slice_1/stack_2?
(model/conv2d_transpose_1/strided_slice_1StridedSlice'model/conv2d_transpose_1/stack:output:07model/conv2d_transpose_1/strided_slice_1/stack:output:09model/conv2d_transpose_1/strided_slice_1/stack_1:output:09model/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(model/conv2d_transpose_1/strided_slice_1?
8model/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpAmodel_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02:
8model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
)model/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput'model/conv2d_transpose_1/stack:output:0@model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0model/add/add:z:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2+
)model/conv2d_transpose_1/conv2d_transpose?
/model/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp8model_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/model/conv2d_transpose_1/BiasAdd/ReadVariableOp?
 model/conv2d_transpose_1/BiasAddBiasAdd2model/conv2d_transpose_1/conv2d_transpose:output:07model/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2"
 model/conv2d_transpose_1/BiasAdd?
model/p_re_lu_4/ReluRelu)model/conv2d_transpose_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/p_re_lu_4/Relu?
model/p_re_lu_4/ReadVariableOpReadVariableOp'model_p_re_lu_4_readvariableop_resource*#
_output_shapes
:?*
dtype02 
model/p_re_lu_4/ReadVariableOp?
model/p_re_lu_4/NegNeg&model/p_re_lu_4/ReadVariableOp:value:0*
T0*#
_output_shapes
:?2
model/p_re_lu_4/Neg?
model/p_re_lu_4/Neg_1Neg)model/conv2d_transpose_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
model/p_re_lu_4/Neg_1?
model/p_re_lu_4/Relu_1Relumodel/p_re_lu_4/Neg_1:y:0*
T0*0
_output_shapes
:??????????2
model/p_re_lu_4/Relu_1?
model/p_re_lu_4/mulMulmodel/p_re_lu_4/Neg:y:0$model/p_re_lu_4/Relu_1:activations:0*
T0*0
_output_shapes
:??????????2
model/p_re_lu_4/mul?
model/p_re_lu_4/addAddV2"model/p_re_lu_4/Relu:activations:0model/p_re_lu_4/mul:z:0*
T0*0
_output_shapes
:??????????2
model/p_re_lu_4/add?
model/conv2d_transpose_2/ShapeShapemodel/p_re_lu_4/add:z:0*
T0*
_output_shapes
:2 
model/conv2d_transpose_2/Shape?
,model/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,model/conv2d_transpose_2/strided_slice/stack?
.model/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv2d_transpose_2/strided_slice/stack_1?
.model/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv2d_transpose_2/strided_slice/stack_2?
&model/conv2d_transpose_2/strided_sliceStridedSlice'model/conv2d_transpose_2/Shape:output:05model/conv2d_transpose_2/strided_slice/stack:output:07model/conv2d_transpose_2/strided_slice/stack_1:output:07model/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&model/conv2d_transpose_2/strided_slice?
 model/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2"
 model/conv2d_transpose_2/stack/1?
 model/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :?2"
 model/conv2d_transpose_2/stack/2?
 model/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2"
 model/conv2d_transpose_2/stack/3?
model/conv2d_transpose_2/stackPack/model/conv2d_transpose_2/strided_slice:output:0)model/conv2d_transpose_2/stack/1:output:0)model/conv2d_transpose_2/stack/2:output:0)model/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2 
model/conv2d_transpose_2/stack?
.model/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model/conv2d_transpose_2/strided_slice_1/stack?
0model/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0model/conv2d_transpose_2/strided_slice_1/stack_1?
0model/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model/conv2d_transpose_2/strided_slice_1/stack_2?
(model/conv2d_transpose_2/strided_slice_1StridedSlice'model/conv2d_transpose_2/stack:output:07model/conv2d_transpose_2/strided_slice_1/stack:output:09model/conv2d_transpose_2/strided_slice_1/stack_1:output:09model/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(model/conv2d_transpose_2/strided_slice_1?
8model/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpAmodel_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02:
8model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
)model/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput'model/conv2d_transpose_2/stack:output:0@model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0model/p_re_lu_4/add:z:0*
T0*0
_output_shapes
:?????????@??*
paddingVALID*
strides
2+
)model/conv2d_transpose_2/conv2d_transpose?
/model/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp8model_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype021
/model/conv2d_transpose_2/BiasAdd/ReadVariableOp?
 model/conv2d_transpose_2/BiasAddBiasAdd2model/conv2d_transpose_2/conv2d_transpose:output:07model/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@??2"
 model/conv2d_transpose_2/BiasAdd?
model/p_re_lu_5/ReluRelu)model/conv2d_transpose_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@??2
model/p_re_lu_5/Relu?
model/p_re_lu_5/ReadVariableOpReadVariableOp'model_p_re_lu_5_readvariableop_resource*#
_output_shapes
:@??*
dtype02 
model/p_re_lu_5/ReadVariableOp?
model/p_re_lu_5/NegNeg&model/p_re_lu_5/ReadVariableOp:value:0*
T0*#
_output_shapes
:@??2
model/p_re_lu_5/Neg?
model/p_re_lu_5/Neg_1Neg)model/conv2d_transpose_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@??2
model/p_re_lu_5/Neg_1?
model/p_re_lu_5/Relu_1Relumodel/p_re_lu_5/Neg_1:y:0*
T0*0
_output_shapes
:?????????@??2
model/p_re_lu_5/Relu_1?
model/p_re_lu_5/mulMulmodel/p_re_lu_5/Neg:y:0$model/p_re_lu_5/Relu_1:activations:0*
T0*0
_output_shapes
:?????????@??2
model/p_re_lu_5/mul?
model/p_re_lu_5/addAddV2"model/p_re_lu_5/Relu:activations:0model/p_re_lu_5/mul:z:0*
T0*0
_output_shapes
:?????????@??2
model/p_re_lu_5/add?
model/conv2d_transpose_3/ShapeShapemodel/p_re_lu_5/add:z:0*
T0*
_output_shapes
:2 
model/conv2d_transpose_3/Shape?
,model/conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,model/conv2d_transpose_3/strided_slice/stack?
.model/conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv2d_transpose_3/strided_slice/stack_1?
.model/conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.model/conv2d_transpose_3/strided_slice/stack_2?
&model/conv2d_transpose_3/strided_sliceStridedSlice'model/conv2d_transpose_3/Shape:output:05model/conv2d_transpose_3/strided_slice/stack:output:07model/conv2d_transpose_3/strided_slice/stack_1:output:07model/conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&model/conv2d_transpose_3/strided_slice?
 model/conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2"
 model/conv2d_transpose_3/stack/1?
 model/conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :?2"
 model/conv2d_transpose_3/stack/2?
 model/conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2"
 model/conv2d_transpose_3/stack/3?
model/conv2d_transpose_3/stackPack/model/conv2d_transpose_3/strided_slice:output:0)model/conv2d_transpose_3/stack/1:output:0)model/conv2d_transpose_3/stack/2:output:0)model/conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2 
model/conv2d_transpose_3/stack?
.model/conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.model/conv2d_transpose_3/strided_slice_1/stack?
0model/conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0model/conv2d_transpose_3/strided_slice_1/stack_1?
0model/conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0model/conv2d_transpose_3/strided_slice_1/stack_2?
(model/conv2d_transpose_3/strided_slice_1StridedSlice'model/conv2d_transpose_3/stack:output:07model/conv2d_transpose_3/strided_slice_1/stack:output:09model/conv2d_transpose_3/strided_slice_1/stack_1:output:09model/conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(model/conv2d_transpose_3/strided_slice_1?
8model/conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOpAmodel_conv2d_transpose_3_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype02:
8model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp?
)model/conv2d_transpose_3/conv2d_transposeConv2DBackpropInput'model/conv2d_transpose_3/stack:output:0@model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0model/p_re_lu_5/add:z:0*
T0*/
_output_shapes
:?????????@?*
paddingSAME*
strides
2+
)model/conv2d_transpose_3/conv2d_transpose?
/model/conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp8model_conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype021
/model/conv2d_transpose_3/BiasAdd/ReadVariableOp?
 model/conv2d_transpose_3/BiasAddBiasAdd2model/conv2d_transpose_3/conv2d_transpose:output:07model/conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?2"
 model/conv2d_transpose_3/BiasAdd?
IdentityIdentity)model/conv2d_transpose_3/BiasAdd:output:0$^model/conv2d/BiasAdd/ReadVariableOp#^model/conv2d/Conv2D/ReadVariableOp&^model/conv2d_1/BiasAdd/ReadVariableOp%^model/conv2d_1/Conv2D/ReadVariableOp&^model/conv2d_2/BiasAdd/ReadVariableOp%^model/conv2d_2/Conv2D/ReadVariableOp.^model/conv2d_transpose/BiasAdd/ReadVariableOp7^model/conv2d_transpose/conv2d_transpose/ReadVariableOp0^model/conv2d_transpose_1/BiasAdd/ReadVariableOp9^model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp0^model/conv2d_transpose_2/BiasAdd/ReadVariableOp9^model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp0^model/conv2d_transpose_3/BiasAdd/ReadVariableOp9^model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp^model/p_re_lu/ReadVariableOp^model/p_re_lu_1/ReadVariableOp^model/p_re_lu_2/ReadVariableOp^model/p_re_lu_3/ReadVariableOp^model/p_re_lu_4/ReadVariableOp^model/p_re_lu_5/ReadVariableOp*
T0*/
_output_shapes
:?????????@?2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????@?::::::::::::::::::::2J
#model/conv2d/BiasAdd/ReadVariableOp#model/conv2d/BiasAdd/ReadVariableOp2H
"model/conv2d/Conv2D/ReadVariableOp"model/conv2d/Conv2D/ReadVariableOp2N
%model/conv2d_1/BiasAdd/ReadVariableOp%model/conv2d_1/BiasAdd/ReadVariableOp2L
$model/conv2d_1/Conv2D/ReadVariableOp$model/conv2d_1/Conv2D/ReadVariableOp2N
%model/conv2d_2/BiasAdd/ReadVariableOp%model/conv2d_2/BiasAdd/ReadVariableOp2L
$model/conv2d_2/Conv2D/ReadVariableOp$model/conv2d_2/Conv2D/ReadVariableOp2^
-model/conv2d_transpose/BiasAdd/ReadVariableOp-model/conv2d_transpose/BiasAdd/ReadVariableOp2p
6model/conv2d_transpose/conv2d_transpose/ReadVariableOp6model/conv2d_transpose/conv2d_transpose/ReadVariableOp2b
/model/conv2d_transpose_1/BiasAdd/ReadVariableOp/model/conv2d_transpose_1/BiasAdd/ReadVariableOp2t
8model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp8model/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2b
/model/conv2d_transpose_2/BiasAdd/ReadVariableOp/model/conv2d_transpose_2/BiasAdd/ReadVariableOp2t
8model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp8model/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2b
/model/conv2d_transpose_3/BiasAdd/ReadVariableOp/model/conv2d_transpose_3/BiasAdd/ReadVariableOp2t
8model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp8model/conv2d_transpose_3/conv2d_transpose/ReadVariableOp2<
model/p_re_lu/ReadVariableOpmodel/p_re_lu/ReadVariableOp2@
model/p_re_lu_1/ReadVariableOpmodel/p_re_lu_1/ReadVariableOp2@
model/p_re_lu_2/ReadVariableOpmodel/p_re_lu_2/ReadVariableOp2@
model/p_re_lu_3/ReadVariableOpmodel/p_re_lu_3/ReadVariableOp2@
model/p_re_lu_4/ReadVariableOpmodel/p_re_lu_4/ReadVariableOp2@
model/p_re_lu_5/ReadVariableOpmodel/p_re_lu_5/ReadVariableOp:X T
/
_output_shapes
:?????????@?
!
_user_specified_name	input_1
?J
?
C__inference_model_layer_call_and_return_conditional_losses_45385315

inputs
conv2d_45385260
conv2d_45385262
p_re_lu_45385265
conv2d_1_45385268
conv2d_1_45385270
p_re_lu_1_45385273
conv2d_2_45385276
conv2d_2_45385278
p_re_lu_2_45385281
conv2d_transpose_45385284
conv2d_transpose_45385286
p_re_lu_3_45385289
conv2d_transpose_1_45385293
conv2d_transpose_1_45385295
p_re_lu_4_45385298
conv2d_transpose_2_45385301
conv2d_transpose_2_45385303
p_re_lu_5_45385306
conv2d_transpose_3_45385309
conv2d_transpose_3_45385311
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?*conv2d_transpose_2/StatefulPartitionedCall?*conv2d_transpose_3/StatefulPartitionedCall?p_re_lu/StatefulPartitionedCall?!p_re_lu_1/StatefulPartitionedCall?!p_re_lu_2/StatefulPartitionedCall?!p_re_lu_3/StatefulPartitionedCall?!p_re_lu_4/StatefulPartitionedCall?!p_re_lu_5/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_45385260conv2d_45385262*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_453850742 
conv2d/StatefulPartitionedCall?
p_re_lu/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0p_re_lu_45385265*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_p_re_lu_layer_call_and_return_conditional_losses_453847592!
p_re_lu/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(p_re_lu/StatefulPartitionedCall:output:0conv2d_1_45385268conv2d_1_45385270*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_453851032"
 conv2d_1/StatefulPartitionedCall?
!p_re_lu_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0p_re_lu_1_45385273*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_1_layer_call_and_return_conditional_losses_453847802#
!p_re_lu_1/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall*p_re_lu_1/StatefulPartitionedCall:output:0conv2d_2_45385276conv2d_2_45385278*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_453851322"
 conv2d_2/StatefulPartitionedCall?
!p_re_lu_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0p_re_lu_2_45385281*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_2_layer_call_and_return_conditional_losses_453848012#
!p_re_lu_2/StatefulPartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall*p_re_lu_2/StatefulPartitionedCall:output:0conv2d_transpose_45385284conv2d_transpose_45385286*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_layer_call_and_return_conditional_losses_453848472*
(conv2d_transpose/StatefulPartitionedCall?
!p_re_lu_3/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0p_re_lu_3_45385289*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_3_layer_call_and_return_conditional_losses_453848702#
!p_re_lu_3/StatefulPartitionedCall?
add/PartitionedCallPartitionedCall*p_re_lu_3/StatefulPartitionedCall:output:0*p_re_lu_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_add_layer_call_and_return_conditional_losses_453851652
add/PartitionedCall?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0conv2d_transpose_1_45385293conv2d_transpose_1_45385295*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_453849162,
*conv2d_transpose_1/StatefulPartitionedCall?
!p_re_lu_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0p_re_lu_4_45385298*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_4_layer_call_and_return_conditional_losses_453849392#
!p_re_lu_4/StatefulPartitionedCall?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall*p_re_lu_4/StatefulPartitionedCall:output:0conv2d_transpose_2_45385301conv2d_transpose_2_45385303*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_453849852,
*conv2d_transpose_2/StatefulPartitionedCall?
!p_re_lu_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0p_re_lu_5_45385306*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@??*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_5_layer_call_and_return_conditional_losses_453850082#
!p_re_lu_5/StatefulPartitionedCall?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall*p_re_lu_5/StatefulPartitionedCall:output:0conv2d_transpose_3_45385309conv2d_transpose_3_45385311*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_453850502,
*conv2d_transpose_3/StatefulPartitionedCall?
IdentityIdentity3conv2d_transpose_3/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall ^p_re_lu/StatefulPartitionedCall"^p_re_lu_1/StatefulPartitionedCall"^p_re_lu_2/StatefulPartitionedCall"^p_re_lu_3/StatefulPartitionedCall"^p_re_lu_4/StatefulPartitionedCall"^p_re_lu_5/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????@?::::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2B
p_re_lu/StatefulPartitionedCallp_re_lu/StatefulPartitionedCall2F
!p_re_lu_1/StatefulPartitionedCall!p_re_lu_1/StatefulPartitionedCall2F
!p_re_lu_2/StatefulPartitionedCall!p_re_lu_2/StatefulPartitionedCall2F
!p_re_lu_3/StatefulPartitionedCall!p_re_lu_3/StatefulPartitionedCall2F
!p_re_lu_4/StatefulPartitionedCall!p_re_lu_4/StatefulPartitionedCall2F
!p_re_lu_5/StatefulPartitionedCall!p_re_lu_5/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@?
 
_user_specified_nameinputs
?J
?
C__inference_model_layer_call_and_return_conditional_losses_45385418

inputs
conv2d_45385363
conv2d_45385365
p_re_lu_45385368
conv2d_1_45385371
conv2d_1_45385373
p_re_lu_1_45385376
conv2d_2_45385379
conv2d_2_45385381
p_re_lu_2_45385384
conv2d_transpose_45385387
conv2d_transpose_45385389
p_re_lu_3_45385392
conv2d_transpose_1_45385396
conv2d_transpose_1_45385398
p_re_lu_4_45385401
conv2d_transpose_2_45385404
conv2d_transpose_2_45385406
p_re_lu_5_45385409
conv2d_transpose_3_45385412
conv2d_transpose_3_45385414
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?*conv2d_transpose_2/StatefulPartitionedCall?*conv2d_transpose_3/StatefulPartitionedCall?p_re_lu/StatefulPartitionedCall?!p_re_lu_1/StatefulPartitionedCall?!p_re_lu_2/StatefulPartitionedCall?!p_re_lu_3/StatefulPartitionedCall?!p_re_lu_4/StatefulPartitionedCall?!p_re_lu_5/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinputsconv2d_45385363conv2d_45385365*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_453850742 
conv2d/StatefulPartitionedCall?
p_re_lu/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0p_re_lu_45385368*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_p_re_lu_layer_call_and_return_conditional_losses_453847592!
p_re_lu/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(p_re_lu/StatefulPartitionedCall:output:0conv2d_1_45385371conv2d_1_45385373*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_453851032"
 conv2d_1/StatefulPartitionedCall?
!p_re_lu_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0p_re_lu_1_45385376*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_1_layer_call_and_return_conditional_losses_453847802#
!p_re_lu_1/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall*p_re_lu_1/StatefulPartitionedCall:output:0conv2d_2_45385379conv2d_2_45385381*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_453851322"
 conv2d_2/StatefulPartitionedCall?
!p_re_lu_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0p_re_lu_2_45385384*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_2_layer_call_and_return_conditional_losses_453848012#
!p_re_lu_2/StatefulPartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall*p_re_lu_2/StatefulPartitionedCall:output:0conv2d_transpose_45385387conv2d_transpose_45385389*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_layer_call_and_return_conditional_losses_453848472*
(conv2d_transpose/StatefulPartitionedCall?
!p_re_lu_3/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0p_re_lu_3_45385392*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_3_layer_call_and_return_conditional_losses_453848702#
!p_re_lu_3/StatefulPartitionedCall?
add/PartitionedCallPartitionedCall*p_re_lu_3/StatefulPartitionedCall:output:0*p_re_lu_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_add_layer_call_and_return_conditional_losses_453851652
add/PartitionedCall?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0conv2d_transpose_1_45385396conv2d_transpose_1_45385398*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_453849162,
*conv2d_transpose_1/StatefulPartitionedCall?
!p_re_lu_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0p_re_lu_4_45385401*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_4_layer_call_and_return_conditional_losses_453849392#
!p_re_lu_4/StatefulPartitionedCall?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall*p_re_lu_4/StatefulPartitionedCall:output:0conv2d_transpose_2_45385404conv2d_transpose_2_45385406*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_453849852,
*conv2d_transpose_2/StatefulPartitionedCall?
!p_re_lu_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0p_re_lu_5_45385409*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@??*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_5_layer_call_and_return_conditional_losses_453850082#
!p_re_lu_5/StatefulPartitionedCall?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall*p_re_lu_5/StatefulPartitionedCall:output:0conv2d_transpose_3_45385412conv2d_transpose_3_45385414*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_453850502,
*conv2d_transpose_3/StatefulPartitionedCall?
IdentityIdentity3conv2d_transpose_3/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall ^p_re_lu/StatefulPartitionedCall"^p_re_lu_1/StatefulPartitionedCall"^p_re_lu_2/StatefulPartitionedCall"^p_re_lu_3/StatefulPartitionedCall"^p_re_lu_4/StatefulPartitionedCall"^p_re_lu_5/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????@?::::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2B
p_re_lu/StatefulPartitionedCallp_re_lu/StatefulPartitionedCall2F
!p_re_lu_1/StatefulPartitionedCall!p_re_lu_1/StatefulPartitionedCall2F
!p_re_lu_2/StatefulPartitionedCall!p_re_lu_2/StatefulPartitionedCall2F
!p_re_lu_3/StatefulPartitionedCall!p_re_lu_3/StatefulPartitionedCall2F
!p_re_lu_4/StatefulPartitionedCall!p_re_lu_4/StatefulPartitionedCall2F
!p_re_lu_5/StatefulPartitionedCall!p_re_lu_5/StatefulPartitionedCall:W S
/
_output_shapes
:?????????@?
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_1_layer_call_and_return_conditional_losses_45385103

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
5__inference_conv2d_transpose_1_layer_call_fn_45384926

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_453849162
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
k
A__inference_add_layer_call_and_return_conditional_losses_45385165

inputs
inputs_1
identity`
addAddV2inputsinputs_1*
T0*0
_output_shapes
:??????????2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????:??????????:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs:XT
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?

?
E__inference_p_re_lu_layer_call_and_return_conditional_losses_45384759

inputs
readvariableop_resource
identity??ReadVariableOpq
ReluReluinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu}
ReadVariableOpReadVariableOpreadvariableop_resource*#
_output_shapes
:?*
dtype02
ReadVariableOpW
NegNegReadVariableOp:value:0*
T0*#
_output_shapes
:?2
Negr
Neg_1Neginputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Neg_1x
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu_1k
mulMulNeg:y:0Relu_1:activations:0*
T0*0
_output_shapes
:??????????2
mulk
addAddV2Relu:activations:0mul:z:0*
T0*0
_output_shapes
:??????????2
addu
IdentityIdentityadd:z:0^ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*M
_input_shapes<
::4????????????????????????????????????:2 
ReadVariableOpReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_model_layer_call_fn_45385358
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_453853152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????@?::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@?
!
_user_specified_name	input_1
?

?
G__inference_p_re_lu_2_layer_call_and_return_conditional_losses_45384801

inputs
readvariableop_resource
identity??ReadVariableOpq
ReluReluinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu}
ReadVariableOpReadVariableOpreadvariableop_resource*#
_output_shapes
:?*
dtype02
ReadVariableOpW
NegNegReadVariableOp:value:0*
T0*#
_output_shapes
:?2
Negr
Neg_1Neginputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Neg_1x
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu_1k
mulMulNeg:y:0Relu_1:activations:0*
T0*0
_output_shapes
:??????????2
mulk
addAddV2Relu:activations:0mul:z:0*
T0*0
_output_shapes
:??????????2
addu
IdentityIdentityadd:z:0^ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*M
_input_shapes<
::4????????????????????????????????????:2 
ReadVariableOpReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?

?
G__inference_p_re_lu_3_layer_call_and_return_conditional_losses_45384870

inputs
readvariableop_resource
identity??ReadVariableOpq
ReluReluinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu}
ReadVariableOpReadVariableOpreadvariableop_resource*#
_output_shapes
:?*
dtype02
ReadVariableOpW
NegNegReadVariableOp:value:0*
T0*#
_output_shapes
:?2
Negr
Neg_1Neginputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Neg_1x
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu_1k
mulMulNeg:y:0Relu_1:activations:0*
T0*0
_output_shapes
:??????????2
mulk
addAddV2Relu:activations:0mul:z:0*
T0*0
_output_shapes
:??????????2
addu
IdentityIdentityadd:z:0^ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*M
_input_shapes<
::4????????????????????????????????????:2 
ReadVariableOpReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
r
,__inference_p_re_lu_1_layer_call_fn_45384788

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_1_layer_call_and_return_conditional_losses_453847802
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*M
_input_shapes<
::4????????????????????????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?&
?
P__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_45384916

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/3?
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_model_layer_call_fn_45385892

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_453854182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????@?::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@?
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_layer_call_and_return_conditional_losses_45385902

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@?
 
_user_specified_nameinputs
?
r
,__inference_p_re_lu_3_layer_call_fn_45384878

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_3_layer_call_and_return_conditional_losses_453848702
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*M
_input_shapes<
::4????????????????????????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
5__inference_conv2d_transpose_3_layer_call_fn_45385060

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_453850502
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

?
G__inference_p_re_lu_4_layer_call_and_return_conditional_losses_45384939

inputs
readvariableop_resource
identity??ReadVariableOpq
ReluReluinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu}
ReadVariableOpReadVariableOpreadvariableop_resource*#
_output_shapes
:?*
dtype02
ReadVariableOpW
NegNegReadVariableOp:value:0*
T0*#
_output_shapes
:?2
Negr
Neg_1Neginputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Neg_1x
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu_1k
mulMulNeg:y:0Relu_1:activations:0*
T0*0
_output_shapes
:??????????2
mulk
addAddV2Relu:activations:0mul:z:0*
T0*0
_output_shapes
:??????????2
addu
IdentityIdentityadd:z:0^ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*M
_input_shapes<
::4????????????????????????????????????:2 
ReadVariableOpReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?&
?
N__inference_conv2d_transpose_layer_call_and_return_conditional_losses_45384847

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/3?
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
R
&__inference_add_layer_call_fn_45385961
inputs_0
inputs_1
identity?
PartitionedCallPartitionedCallinputs_0inputs_1*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_add_layer_call_and_return_conditional_losses_453851652
PartitionedCallu
IdentityIdentityPartitionedCall:output:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????:??????????:Z V
0
_output_shapes
:??????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:??????????
"
_user_specified_name
inputs/1
?	
?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_45385940

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
r
,__inference_p_re_lu_2_layer_call_fn_45384809

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_2_layer_call_and_return_conditional_losses_453848012
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*M
_input_shapes<
::4????????????????????????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
(__inference_model_layer_call_fn_45385461
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_453854182
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????@?::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@?
!
_user_specified_name	input_1
?

?
G__inference_p_re_lu_5_layer_call_and_return_conditional_losses_45385008

inputs
readvariableop_resource
identity??ReadVariableOpq
ReluReluinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu}
ReadVariableOpReadVariableOpreadvariableop_resource*#
_output_shapes
:@??*
dtype02
ReadVariableOpW
NegNegReadVariableOp:value:0*
T0*#
_output_shapes
:@??2
Negr
Neg_1Neginputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Neg_1x
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu_1k
mulMulNeg:y:0Relu_1:activations:0*
T0*0
_output_shapes
:?????????@??2
mulk
addAddV2Relu:activations:0mul:z:0*
T0*0
_output_shapes
:?????????@??2
addu
IdentityIdentityadd:z:0^ReadVariableOp*
T0*0
_output_shapes
:?????????@??2

Identity"
identityIdentity:output:0*M
_input_shapes<
::4????????????????????????????????????:2 
ReadVariableOpReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_45385132

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
&__inference_signature_wrapper_45385508
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????@?*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference__wrapped_model_453847462
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????@?2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????@?::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????@?
!
_user_specified_name	input_1
?#
?
P__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_45385050

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?J
?
C__inference_model_layer_call_and_return_conditional_losses_45385196
input_1
conv2d_45385085
conv2d_45385087
p_re_lu_45385090
conv2d_1_45385114
conv2d_1_45385116
p_re_lu_1_45385119
conv2d_2_45385143
conv2d_2_45385145
p_re_lu_2_45385148
conv2d_transpose_45385151
conv2d_transpose_45385153
p_re_lu_3_45385156
conv2d_transpose_1_45385174
conv2d_transpose_1_45385176
p_re_lu_4_45385179
conv2d_transpose_2_45385182
conv2d_transpose_2_45385184
p_re_lu_5_45385187
conv2d_transpose_3_45385190
conv2d_transpose_3_45385192
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?*conv2d_transpose_2/StatefulPartitionedCall?*conv2d_transpose_3/StatefulPartitionedCall?p_re_lu/StatefulPartitionedCall?!p_re_lu_1/StatefulPartitionedCall?!p_re_lu_2/StatefulPartitionedCall?!p_re_lu_3/StatefulPartitionedCall?!p_re_lu_4/StatefulPartitionedCall?!p_re_lu_5/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_45385085conv2d_45385087*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_453850742 
conv2d/StatefulPartitionedCall?
p_re_lu/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0p_re_lu_45385090*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_p_re_lu_layer_call_and_return_conditional_losses_453847592!
p_re_lu/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(p_re_lu/StatefulPartitionedCall:output:0conv2d_1_45385114conv2d_1_45385116*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_453851032"
 conv2d_1/StatefulPartitionedCall?
!p_re_lu_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0p_re_lu_1_45385119*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_1_layer_call_and_return_conditional_losses_453847802#
!p_re_lu_1/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall*p_re_lu_1/StatefulPartitionedCall:output:0conv2d_2_45385143conv2d_2_45385145*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_453851322"
 conv2d_2/StatefulPartitionedCall?
!p_re_lu_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0p_re_lu_2_45385148*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_2_layer_call_and_return_conditional_losses_453848012#
!p_re_lu_2/StatefulPartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall*p_re_lu_2/StatefulPartitionedCall:output:0conv2d_transpose_45385151conv2d_transpose_45385153*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_layer_call_and_return_conditional_losses_453848472*
(conv2d_transpose/StatefulPartitionedCall?
!p_re_lu_3/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0p_re_lu_3_45385156*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_3_layer_call_and_return_conditional_losses_453848702#
!p_re_lu_3/StatefulPartitionedCall?
add/PartitionedCallPartitionedCall*p_re_lu_3/StatefulPartitionedCall:output:0*p_re_lu_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_add_layer_call_and_return_conditional_losses_453851652
add/PartitionedCall?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0conv2d_transpose_1_45385174conv2d_transpose_1_45385176*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_453849162,
*conv2d_transpose_1/StatefulPartitionedCall?
!p_re_lu_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0p_re_lu_4_45385179*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_4_layer_call_and_return_conditional_losses_453849392#
!p_re_lu_4/StatefulPartitionedCall?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall*p_re_lu_4/StatefulPartitionedCall:output:0conv2d_transpose_2_45385182conv2d_transpose_2_45385184*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_453849852,
*conv2d_transpose_2/StatefulPartitionedCall?
!p_re_lu_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0p_re_lu_5_45385187*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@??*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_5_layer_call_and_return_conditional_losses_453850082#
!p_re_lu_5/StatefulPartitionedCall?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall*p_re_lu_5/StatefulPartitionedCall:output:0conv2d_transpose_3_45385190conv2d_transpose_3_45385192*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_453850502,
*conv2d_transpose_3/StatefulPartitionedCall?
IdentityIdentity3conv2d_transpose_3/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall ^p_re_lu/StatefulPartitionedCall"^p_re_lu_1/StatefulPartitionedCall"^p_re_lu_2/StatefulPartitionedCall"^p_re_lu_3/StatefulPartitionedCall"^p_re_lu_4/StatefulPartitionedCall"^p_re_lu_5/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????@?::::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2B
p_re_lu/StatefulPartitionedCallp_re_lu/StatefulPartitionedCall2F
!p_re_lu_1/StatefulPartitionedCall!p_re_lu_1/StatefulPartitionedCall2F
!p_re_lu_2/StatefulPartitionedCall!p_re_lu_2/StatefulPartitionedCall2F
!p_re_lu_3/StatefulPartitionedCall!p_re_lu_3/StatefulPartitionedCall2F
!p_re_lu_4/StatefulPartitionedCall!p_re_lu_4/StatefulPartitionedCall2F
!p_re_lu_5/StatefulPartitionedCall!p_re_lu_5/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@?
!
_user_specified_name	input_1
?J
?
C__inference_model_layer_call_and_return_conditional_losses_45385254
input_1
conv2d_45385199
conv2d_45385201
p_re_lu_45385204
conv2d_1_45385207
conv2d_1_45385209
p_re_lu_1_45385212
conv2d_2_45385215
conv2d_2_45385217
p_re_lu_2_45385220
conv2d_transpose_45385223
conv2d_transpose_45385225
p_re_lu_3_45385228
conv2d_transpose_1_45385232
conv2d_transpose_1_45385234
p_re_lu_4_45385237
conv2d_transpose_2_45385240
conv2d_transpose_2_45385242
p_re_lu_5_45385245
conv2d_transpose_3_45385248
conv2d_transpose_3_45385250
identity??conv2d/StatefulPartitionedCall? conv2d_1/StatefulPartitionedCall? conv2d_2/StatefulPartitionedCall?(conv2d_transpose/StatefulPartitionedCall?*conv2d_transpose_1/StatefulPartitionedCall?*conv2d_transpose_2/StatefulPartitionedCall?*conv2d_transpose_3/StatefulPartitionedCall?p_re_lu/StatefulPartitionedCall?!p_re_lu_1/StatefulPartitionedCall?!p_re_lu_2/StatefulPartitionedCall?!p_re_lu_3/StatefulPartitionedCall?!p_re_lu_4/StatefulPartitionedCall?!p_re_lu_5/StatefulPartitionedCall?
conv2d/StatefulPartitionedCallStatefulPartitionedCallinput_1conv2d_45385199conv2d_45385201*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_453850742 
conv2d/StatefulPartitionedCall?
p_re_lu/StatefulPartitionedCallStatefulPartitionedCall'conv2d/StatefulPartitionedCall:output:0p_re_lu_45385204*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_p_re_lu_layer_call_and_return_conditional_losses_453847592!
p_re_lu/StatefulPartitionedCall?
 conv2d_1/StatefulPartitionedCallStatefulPartitionedCall(p_re_lu/StatefulPartitionedCall:output:0conv2d_1_45385207conv2d_1_45385209*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_453851032"
 conv2d_1/StatefulPartitionedCall?
!p_re_lu_1/StatefulPartitionedCallStatefulPartitionedCall)conv2d_1/StatefulPartitionedCall:output:0p_re_lu_1_45385212*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_1_layer_call_and_return_conditional_losses_453847802#
!p_re_lu_1/StatefulPartitionedCall?
 conv2d_2/StatefulPartitionedCallStatefulPartitionedCall*p_re_lu_1/StatefulPartitionedCall:output:0conv2d_2_45385215conv2d_2_45385217*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_453851322"
 conv2d_2/StatefulPartitionedCall?
!p_re_lu_2/StatefulPartitionedCallStatefulPartitionedCall)conv2d_2/StatefulPartitionedCall:output:0p_re_lu_2_45385220*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_2_layer_call_and_return_conditional_losses_453848012#
!p_re_lu_2/StatefulPartitionedCall?
(conv2d_transpose/StatefulPartitionedCallStatefulPartitionedCall*p_re_lu_2/StatefulPartitionedCall:output:0conv2d_transpose_45385223conv2d_transpose_45385225*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_layer_call_and_return_conditional_losses_453848472*
(conv2d_transpose/StatefulPartitionedCall?
!p_re_lu_3/StatefulPartitionedCallStatefulPartitionedCall1conv2d_transpose/StatefulPartitionedCall:output:0p_re_lu_3_45385228*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_3_layer_call_and_return_conditional_losses_453848702#
!p_re_lu_3/StatefulPartitionedCall?
add/PartitionedCallPartitionedCall*p_re_lu_3/StatefulPartitionedCall:output:0*p_re_lu_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *J
fERC
A__inference_add_layer_call_and_return_conditional_losses_453851652
add/PartitionedCall?
*conv2d_transpose_1/StatefulPartitionedCallStatefulPartitionedCalladd/PartitionedCall:output:0conv2d_transpose_1_45385232conv2d_transpose_1_45385234*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_453849162,
*conv2d_transpose_1/StatefulPartitionedCall?
!p_re_lu_4/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_1/StatefulPartitionedCall:output:0p_re_lu_4_45385237*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_4_layer_call_and_return_conditional_losses_453849392#
!p_re_lu_4/StatefulPartitionedCall?
*conv2d_transpose_2/StatefulPartitionedCallStatefulPartitionedCall*p_re_lu_4/StatefulPartitionedCall:output:0conv2d_transpose_2_45385240conv2d_transpose_2_45385242*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_453849852,
*conv2d_transpose_2/StatefulPartitionedCall?
!p_re_lu_5/StatefulPartitionedCallStatefulPartitionedCall3conv2d_transpose_2/StatefulPartitionedCall:output:0p_re_lu_5_45385245*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@??*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_5_layer_call_and_return_conditional_losses_453850082#
!p_re_lu_5/StatefulPartitionedCall?
*conv2d_transpose_3/StatefulPartitionedCallStatefulPartitionedCall*p_re_lu_5/StatefulPartitionedCall:output:0conv2d_transpose_3_45385248conv2d_transpose_3_45385250*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_453850502,
*conv2d_transpose_3/StatefulPartitionedCall?
IdentityIdentity3conv2d_transpose_3/StatefulPartitionedCall:output:0^conv2d/StatefulPartitionedCall!^conv2d_1/StatefulPartitionedCall!^conv2d_2/StatefulPartitionedCall)^conv2d_transpose/StatefulPartitionedCall+^conv2d_transpose_1/StatefulPartitionedCall+^conv2d_transpose_2/StatefulPartitionedCall+^conv2d_transpose_3/StatefulPartitionedCall ^p_re_lu/StatefulPartitionedCall"^p_re_lu_1/StatefulPartitionedCall"^p_re_lu_2/StatefulPartitionedCall"^p_re_lu_3/StatefulPartitionedCall"^p_re_lu_4/StatefulPartitionedCall"^p_re_lu_5/StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????@?::::::::::::::::::::2@
conv2d/StatefulPartitionedCallconv2d/StatefulPartitionedCall2D
 conv2d_1/StatefulPartitionedCall conv2d_1/StatefulPartitionedCall2D
 conv2d_2/StatefulPartitionedCall conv2d_2/StatefulPartitionedCall2T
(conv2d_transpose/StatefulPartitionedCall(conv2d_transpose/StatefulPartitionedCall2X
*conv2d_transpose_1/StatefulPartitionedCall*conv2d_transpose_1/StatefulPartitionedCall2X
*conv2d_transpose_2/StatefulPartitionedCall*conv2d_transpose_2/StatefulPartitionedCall2X
*conv2d_transpose_3/StatefulPartitionedCall*conv2d_transpose_3/StatefulPartitionedCall2B
p_re_lu/StatefulPartitionedCallp_re_lu/StatefulPartitionedCall2F
!p_re_lu_1/StatefulPartitionedCall!p_re_lu_1/StatefulPartitionedCall2F
!p_re_lu_2/StatefulPartitionedCall!p_re_lu_2/StatefulPartitionedCall2F
!p_re_lu_3/StatefulPartitionedCall!p_re_lu_3/StatefulPartitionedCall2F
!p_re_lu_4/StatefulPartitionedCall!p_re_lu_4/StatefulPartitionedCall2F
!p_re_lu_5/StatefulPartitionedCall!p_re_lu_5/StatefulPartitionedCall:X T
/
_output_shapes
:?????????@?
!
_user_specified_name	input_1
??
?
C__inference_model_layer_call_and_return_conditional_losses_45385655

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource#
p_re_lu_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource%
!p_re_lu_1_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource%
!p_re_lu_2_readvariableop_resource=
9conv2d_transpose_conv2d_transpose_readvariableop_resource4
0conv2d_transpose_biasadd_readvariableop_resource%
!p_re_lu_3_readvariableop_resource?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource%
!p_re_lu_4_readvariableop_resource?
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_2_biasadd_readvariableop_resource%
!p_re_lu_5_readvariableop_resource?
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_3_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?)conv2d_transpose_2/BiasAdd/ReadVariableOp?2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?)conv2d_transpose_3/BiasAdd/ReadVariableOp?2conv2d_transpose_3/conv2d_transpose/ReadVariableOp?p_re_lu/ReadVariableOp?p_re_lu_1/ReadVariableOp?p_re_lu_2/ReadVariableOp?p_re_lu_3/ReadVariableOp?p_re_lu_4/ReadVariableOp?p_re_lu_5/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d/BiasAddx
p_re_lu/ReluReluconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
p_re_lu/Relu?
p_re_lu/ReadVariableOpReadVariableOpp_re_lu_readvariableop_resource*#
_output_shapes
:?*
dtype02
p_re_lu/ReadVariableOpo
p_re_lu/NegNegp_re_lu/ReadVariableOp:value:0*
T0*#
_output_shapes
:?2
p_re_lu/Negy
p_re_lu/Neg_1Negconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
p_re_lu/Neg_1v
p_re_lu/Relu_1Relup_re_lu/Neg_1:y:0*
T0*0
_output_shapes
:??????????2
p_re_lu/Relu_1?
p_re_lu/mulMulp_re_lu/Neg:y:0p_re_lu/Relu_1:activations:0*
T0*0
_output_shapes
:??????????2
p_re_lu/mul?
p_re_lu/addAddV2p_re_lu/Relu:activations:0p_re_lu/mul:z:0*
T0*0
_output_shapes
:??????????2
p_re_lu/add?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dp_re_lu/add:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_1/BiasAdd~
p_re_lu_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
p_re_lu_1/Relu?
p_re_lu_1/ReadVariableOpReadVariableOp!p_re_lu_1_readvariableop_resource*#
_output_shapes
:?*
dtype02
p_re_lu_1/ReadVariableOpu
p_re_lu_1/NegNeg p_re_lu_1/ReadVariableOp:value:0*
T0*#
_output_shapes
:?2
p_re_lu_1/Neg
p_re_lu_1/Neg_1Negconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
p_re_lu_1/Neg_1|
p_re_lu_1/Relu_1Relup_re_lu_1/Neg_1:y:0*
T0*0
_output_shapes
:??????????2
p_re_lu_1/Relu_1?
p_re_lu_1/mulMulp_re_lu_1/Neg:y:0p_re_lu_1/Relu_1:activations:0*
T0*0
_output_shapes
:??????????2
p_re_lu_1/mul?
p_re_lu_1/addAddV2p_re_lu_1/Relu:activations:0p_re_lu_1/mul:z:0*
T0*0
_output_shapes
:??????????2
p_re_lu_1/add?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dp_re_lu_1/add:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_2/BiasAdd~
p_re_lu_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
p_re_lu_2/Relu?
p_re_lu_2/ReadVariableOpReadVariableOp!p_re_lu_2_readvariableop_resource*#
_output_shapes
:?*
dtype02
p_re_lu_2/ReadVariableOpu
p_re_lu_2/NegNeg p_re_lu_2/ReadVariableOp:value:0*
T0*#
_output_shapes
:?2
p_re_lu_2/Neg
p_re_lu_2/Neg_1Negconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
p_re_lu_2/Neg_1|
p_re_lu_2/Relu_1Relup_re_lu_2/Neg_1:y:0*
T0*0
_output_shapes
:??????????2
p_re_lu_2/Relu_1?
p_re_lu_2/mulMulp_re_lu_2/Neg:y:0p_re_lu_2/Relu_1:activations:0*
T0*0
_output_shapes
:??????????2
p_re_lu_2/mul?
p_re_lu_2/addAddV2p_re_lu_2/Relu:activations:0p_re_lu_2/mul:z:0*
T0*0
_output_shapes
:??????????2
p_re_lu_2/addq
conv2d_transpose/ShapeShapep_re_lu_2/add:z:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicev
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/1v
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/2w
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0p_re_lu_2/add:z:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_transpose/BiasAdd?
p_re_lu_3/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
p_re_lu_3/Relu?
p_re_lu_3/ReadVariableOpReadVariableOp!p_re_lu_3_readvariableop_resource*#
_output_shapes
:?*
dtype02
p_re_lu_3/ReadVariableOpu
p_re_lu_3/NegNeg p_re_lu_3/ReadVariableOp:value:0*
T0*#
_output_shapes
:?2
p_re_lu_3/Neg?
p_re_lu_3/Neg_1Neg!conv2d_transpose/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
p_re_lu_3/Neg_1|
p_re_lu_3/Relu_1Relup_re_lu_3/Neg_1:y:0*
T0*0
_output_shapes
:??????????2
p_re_lu_3/Relu_1?
p_re_lu_3/mulMulp_re_lu_3/Neg:y:0p_re_lu_3/Relu_1:activations:0*
T0*0
_output_shapes
:??????????2
p_re_lu_3/mul?
p_re_lu_3/addAddV2p_re_lu_3/Relu:activations:0p_re_lu_3/mul:z:0*
T0*0
_output_shapes
:??????????2
p_re_lu_3/add|
add/addAddV2p_re_lu_3/add:z:0p_re_lu_1/add:z:0*
T0*0
_output_shapes
:??????????2	
add/addo
conv2d_transpose_1/ShapeShapeadd/add:z:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape?
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack?
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1?
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slicez
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/1z
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/2{
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_1/stack/3?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack?
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stack?
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1?
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0add/add:z:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2%
#conv2d_transpose_1/conv2d_transpose?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOp?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_transpose_1/BiasAdd?
p_re_lu_4/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
p_re_lu_4/Relu?
p_re_lu_4/ReadVariableOpReadVariableOp!p_re_lu_4_readvariableop_resource*#
_output_shapes
:?*
dtype02
p_re_lu_4/ReadVariableOpu
p_re_lu_4/NegNeg p_re_lu_4/ReadVariableOp:value:0*
T0*#
_output_shapes
:?2
p_re_lu_4/Neg?
p_re_lu_4/Neg_1Neg#conv2d_transpose_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
p_re_lu_4/Neg_1|
p_re_lu_4/Relu_1Relup_re_lu_4/Neg_1:y:0*
T0*0
_output_shapes
:??????????2
p_re_lu_4/Relu_1?
p_re_lu_4/mulMulp_re_lu_4/Neg:y:0p_re_lu_4/Relu_1:activations:0*
T0*0
_output_shapes
:??????????2
p_re_lu_4/mul?
p_re_lu_4/addAddV2p_re_lu_4/Relu:activations:0p_re_lu_4/mul:z:0*
T0*0
_output_shapes
:??????????2
p_re_lu_4/addu
conv2d_transpose_2/ShapeShapep_re_lu_4/add:z:0*
T0*
_output_shapes
:2
conv2d_transpose_2/Shape?
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_2/strided_slice/stack?
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_1?
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_2?
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_2/strided_slicez
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_2/stack/1z
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :?2
conv2d_transpose_2/stack/2{
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_2/stack/3?
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_2/stack?
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_2/strided_slice_1/stack?
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_1?
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_2?
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_1?
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype024
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0p_re_lu_4/add:z:0*
T0*0
_output_shapes
:?????????@??*
paddingVALID*
strides
2%
#conv2d_transpose_2/conv2d_transpose?
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)conv2d_transpose_2/BiasAdd/ReadVariableOp?
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@??2
conv2d_transpose_2/BiasAdd?
p_re_lu_5/ReluRelu#conv2d_transpose_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@??2
p_re_lu_5/Relu?
p_re_lu_5/ReadVariableOpReadVariableOp!p_re_lu_5_readvariableop_resource*#
_output_shapes
:@??*
dtype02
p_re_lu_5/ReadVariableOpu
p_re_lu_5/NegNeg p_re_lu_5/ReadVariableOp:value:0*
T0*#
_output_shapes
:@??2
p_re_lu_5/Neg?
p_re_lu_5/Neg_1Neg#conv2d_transpose_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@??2
p_re_lu_5/Neg_1|
p_re_lu_5/Relu_1Relup_re_lu_5/Neg_1:y:0*
T0*0
_output_shapes
:?????????@??2
p_re_lu_5/Relu_1?
p_re_lu_5/mulMulp_re_lu_5/Neg:y:0p_re_lu_5/Relu_1:activations:0*
T0*0
_output_shapes
:?????????@??2
p_re_lu_5/mul?
p_re_lu_5/addAddV2p_re_lu_5/Relu:activations:0p_re_lu_5/mul:z:0*
T0*0
_output_shapes
:?????????@??2
p_re_lu_5/addu
conv2d_transpose_3/ShapeShapep_re_lu_5/add:z:0*
T0*
_output_shapes
:2
conv2d_transpose_3/Shape?
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_3/strided_slice/stack?
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_1?
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_2?
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_3/strided_slicez
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_3/stack/1z
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :?2
conv2d_transpose_3/stack/2z
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_3/stack/3?
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_3/stack?
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_3/strided_slice_1/stack?
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_1?
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_2?
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_1?
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype024
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0p_re_lu_5/add:z:0*
T0*/
_output_shapes
:?????????@?*
paddingSAME*
strides
2%
#conv2d_transpose_3/conv2d_transpose?
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_3/BiasAdd/ReadVariableOp?
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?2
conv2d_transpose_3/BiasAdd?
IdentityIdentity#conv2d_transpose_3/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp^p_re_lu/ReadVariableOp^p_re_lu_1/ReadVariableOp^p_re_lu_2/ReadVariableOp^p_re_lu_3/ReadVariableOp^p_re_lu_4/ReadVariableOp^p_re_lu_5/ReadVariableOp*
T0*/
_output_shapes
:?????????@?2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????@?::::::::::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp20
p_re_lu/ReadVariableOpp_re_lu/ReadVariableOp24
p_re_lu_1/ReadVariableOpp_re_lu_1/ReadVariableOp24
p_re_lu_2/ReadVariableOpp_re_lu_2/ReadVariableOp24
p_re_lu_3/ReadVariableOpp_re_lu_3/ReadVariableOp24
p_re_lu_4/ReadVariableOpp_re_lu_4/ReadVariableOp24
p_re_lu_5/ReadVariableOpp_re_lu_5/ReadVariableOp:W S
/
_output_shapes
:?????????@?
 
_user_specified_nameinputs
?
?
5__inference_conv2d_transpose_2_layer_call_fn_45384995

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *Y
fTRR
P__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_453849852
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?

?
G__inference_p_re_lu_1_layer_call_and_return_conditional_losses_45384780

inputs
readvariableop_resource
identity??ReadVariableOpq
ReluReluinputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu}
ReadVariableOpReadVariableOpreadvariableop_resource*#
_output_shapes
:?*
dtype02
ReadVariableOpW
NegNegReadVariableOp:value:0*
T0*#
_output_shapes
:?2
Negr
Neg_1Neginputs*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Neg_1x
Relu_1Relu	Neg_1:y:0*
T0*J
_output_shapes8
6:4????????????????????????????????????2
Relu_1k
mulMulNeg:y:0Relu_1:activations:0*
T0*0
_output_shapes
:??????????2
mulk
addAddV2Relu:activations:0mul:z:0*
T0*0
_output_shapes
:??????????2
addu
IdentityIdentityadd:z:0^ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*M
_input_shapes<
::4????????????????????????????????????:2 
ReadVariableOpReadVariableOp:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
~
)__inference_conv2d_layer_call_fn_45385911

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *M
fHRF
D__inference_conv2d_layer_call_and_return_conditional_losses_453850742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@?::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@?
 
_user_specified_nameinputs
?	
?
D__inference_conv2d_layer_call_and_return_conditional_losses_45385074

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*6
_input_shapes%
#:?????????@?::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:W S
/
_output_shapes
:?????????@?
 
_user_specified_nameinputs
?&
?
P__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_45384985

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulP
add/yConst*
_output_shapes
: *
dtype0*
value	B :2
add/yM
addAddV2mul:z:0add/y:output:0*
T0*
_output_shapes
: 2
addT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
add_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
add_1/yU
add_1AddV2	mul_1:z:0add_1/y:output:0*
T0*
_output_shapes
: 2
add_1U
stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2	
stack/3?
stackPackstrided_slice:output:0add:z:0	add_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*B
_output_shapes0
.:,????????????????????????????*
paddingVALID*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*B
_output_shapes0
.:,????????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?	
?
F__inference_conv2d_1_layer_call_and_return_conditional_losses_45385921

inputs"
conv2d_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?Conv2D/ReadVariableOp?
Conv2D/ReadVariableOpReadVariableOpconv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02
Conv2D/ReadVariableOp?
Conv2DConv2DinputsConv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
Conv2D?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddConv2D:output:0BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^Conv2D/ReadVariableOp*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
Conv2D/ReadVariableOpConv2D/ReadVariableOp:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
3__inference_conv2d_transpose_layer_call_fn_45384857

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *B
_output_shapes0
.:,????????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *W
fRRP
N__inference_conv2d_transpose_layer_call_and_return_conditional_losses_453848472
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*B
_output_shapes0
.:,????????????????????????????2

Identity"
identityIdentity:output:0*I
_input_shapes8
6:,????????????????????????????::22
StatefulPartitionedCallStatefulPartitionedCall:j f
B
_output_shapes0
.:,????????????????????????????
 
_user_specified_nameinputs
?
r
,__inference_p_re_lu_5_layer_call_fn_45385016

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:?????????@??*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_5_layer_call_and_return_conditional_losses_453850082
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:?????????@??2

Identity"
identityIdentity:output:0*M
_input_shapes<
::4????????????????????????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
p
*__inference_p_re_lu_layer_call_fn_45384767

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *N
fIRG
E__inference_p_re_lu_layer_call_and_return_conditional_losses_453847592
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*M
_input_shapes<
::4????????????????????????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_conv2d_1_layer_call_fn_45385930

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_1_layer_call_and_return_conditional_losses_453851032
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
r
,__inference_p_re_lu_4_layer_call_fn_45384947

inputs
unknown
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*#
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *P
fKRI
G__inference_p_re_lu_4_layer_call_and_return_conditional_losses_453849392
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*M
_input_shapes<
::4????????????????????????????????????:22
StatefulPartitionedCallStatefulPartitionedCall:r n
J
_output_shapes8
6:4????????????????????????????????????
 
_user_specified_nameinputs
?
?
+__inference_conv2d_2_layer_call_fn_45385949

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *0
_output_shapes
:??????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *O
fJRH
F__inference_conv2d_2_layer_call_and_return_conditional_losses_453851322
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:??????????::22
StatefulPartitionedCallStatefulPartitionedCall:X T
0
_output_shapes
:??????????
 
_user_specified_nameinputs
?
?
(__inference_model_layer_call_fn_45385847

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14

unknown_15

unknown_16

unknown_17

unknown_18
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14
unknown_15
unknown_16
unknown_17
unknown_18* 
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*6
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *L
fGRE
C__inference_model_layer_call_and_return_conditional_losses_453853152
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????@?::::::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????@?
 
_user_specified_nameinputs
?3
?
!__inference__traced_save_45386044
file_prefix,
(savev2_conv2d_kernel_read_readvariableop*
&savev2_conv2d_bias_read_readvariableop,
(savev2_p_re_lu_alpha_read_readvariableop.
*savev2_conv2d_1_kernel_read_readvariableop,
(savev2_conv2d_1_bias_read_readvariableop.
*savev2_p_re_lu_1_alpha_read_readvariableop.
*savev2_conv2d_2_kernel_read_readvariableop,
(savev2_conv2d_2_bias_read_readvariableop.
*savev2_p_re_lu_2_alpha_read_readvariableop6
2savev2_conv2d_transpose_kernel_read_readvariableop4
0savev2_conv2d_transpose_bias_read_readvariableop.
*savev2_p_re_lu_3_alpha_read_readvariableop8
4savev2_conv2d_transpose_1_kernel_read_readvariableop6
2savev2_conv2d_transpose_1_bias_read_readvariableop.
*savev2_p_re_lu_4_alpha_read_readvariableop8
4savev2_conv2d_transpose_2_kernel_read_readvariableop6
2savev2_conv2d_transpose_2_bias_read_readvariableop.
*savev2_p_re_lu_5_alpha_read_readvariableop8
4savev2_conv2d_transpose_3_kernel_read_readvariableop6
2savev2_conv2d_transpose_3_bias_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?	
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*?	
value?B?B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-1/alpha/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-3/alpha/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-4/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-4/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-5/alpha/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-6/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-6/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-7/alpha/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-8/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-8/bias/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-9/alpha/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-10/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-10/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-11/alpha/.ATTRIBUTES/VARIABLE_VALUEB7layer_with_weights-12/kernel/.ATTRIBUTES/VARIABLE_VALUEB5layer_with_weights-12/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*=
value4B2B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices?
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0(savev2_conv2d_kernel_read_readvariableop&savev2_conv2d_bias_read_readvariableop(savev2_p_re_lu_alpha_read_readvariableop*savev2_conv2d_1_kernel_read_readvariableop(savev2_conv2d_1_bias_read_readvariableop*savev2_p_re_lu_1_alpha_read_readvariableop*savev2_conv2d_2_kernel_read_readvariableop(savev2_conv2d_2_bias_read_readvariableop*savev2_p_re_lu_2_alpha_read_readvariableop2savev2_conv2d_transpose_kernel_read_readvariableop0savev2_conv2d_transpose_bias_read_readvariableop*savev2_p_re_lu_3_alpha_read_readvariableop4savev2_conv2d_transpose_1_kernel_read_readvariableop2savev2_conv2d_transpose_1_bias_read_readvariableop*savev2_p_re_lu_4_alpha_read_readvariableop4savev2_conv2d_transpose_2_kernel_read_readvariableop2savev2_conv2d_transpose_2_bias_read_readvariableop*savev2_p_re_lu_5_alpha_read_readvariableop4savev2_conv2d_transpose_3_kernel_read_readvariableop2savev2_conv2d_transpose_3_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *#
dtypes
22
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: :?:?:?:??:?:?:??:?:?:??:?:?:??:?:?:??:?:@??:?:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:-)
'
_output_shapes
:?:!

_output_shapes	
:?:)%
#
_output_shapes
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:)%
#
_output_shapes
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:)	%
#
_output_shapes
:?:.
*
(
_output_shapes
:??:!

_output_shapes	
:?:)%
#
_output_shapes
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:)%
#
_output_shapes
:?:.*
(
_output_shapes
:??:!

_output_shapes	
:?:)%
#
_output_shapes
:@??:-)
'
_output_shapes
:?: 

_output_shapes
::

_output_shapes
: 
?
m
A__inference_add_layer_call_and_return_conditional_losses_45385955
inputs_0
inputs_1
identityb
addAddV2inputs_0inputs_1*
T0*0
_output_shapes
:??????????2
addd
IdentityIdentityadd:z:0*
T0*0
_output_shapes
:??????????2

Identity"
identityIdentity:output:0*K
_input_shapes:
8:??????????:??????????:Z V
0
_output_shapes
:??????????
"
_user_specified_name
inputs/0:ZV
0
_output_shapes
:??????????
"
_user_specified_name
inputs/1
??
?
C__inference_model_layer_call_and_return_conditional_losses_45385802

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource#
p_re_lu_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource%
!p_re_lu_1_readvariableop_resource+
'conv2d_2_conv2d_readvariableop_resource,
(conv2d_2_biasadd_readvariableop_resource%
!p_re_lu_2_readvariableop_resource=
9conv2d_transpose_conv2d_transpose_readvariableop_resource4
0conv2d_transpose_biasadd_readvariableop_resource%
!p_re_lu_3_readvariableop_resource?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource%
!p_re_lu_4_readvariableop_resource?
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_2_biasadd_readvariableop_resource%
!p_re_lu_5_readvariableop_resource?
;conv2d_transpose_3_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_3_biasadd_readvariableop_resource
identity??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?conv2d_2/BiasAdd/ReadVariableOp?conv2d_2/Conv2D/ReadVariableOp?'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?)conv2d_transpose_2/BiasAdd/ReadVariableOp?2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?)conv2d_transpose_3/BiasAdd/ReadVariableOp?2conv2d_transpose_3/conv2d_transpose/ReadVariableOp?p_re_lu/ReadVariableOp?p_re_lu_1/ReadVariableOp?p_re_lu_2/ReadVariableOp?p_re_lu_3/ReadVariableOp?p_re_lu_4/ReadVariableOp?p_re_lu_5/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*'
_output_shapes
:?*
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d/BiasAddx
p_re_lu/ReluReluconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
p_re_lu/Relu?
p_re_lu/ReadVariableOpReadVariableOpp_re_lu_readvariableop_resource*#
_output_shapes
:?*
dtype02
p_re_lu/ReadVariableOpo
p_re_lu/NegNegp_re_lu/ReadVariableOp:value:0*
T0*#
_output_shapes
:?2
p_re_lu/Negy
p_re_lu/Neg_1Negconv2d/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
p_re_lu/Neg_1v
p_re_lu/Relu_1Relup_re_lu/Neg_1:y:0*
T0*0
_output_shapes
:??????????2
p_re_lu/Relu_1?
p_re_lu/mulMulp_re_lu/Neg:y:0p_re_lu/Relu_1:activations:0*
T0*0
_output_shapes
:??????????2
p_re_lu/mul?
p_re_lu/addAddV2p_re_lu/Relu:activations:0p_re_lu/mul:z:0*
T0*0
_output_shapes
:??????????2
p_re_lu/add?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dp_re_lu/add:z:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_1/BiasAdd~
p_re_lu_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
p_re_lu_1/Relu?
p_re_lu_1/ReadVariableOpReadVariableOp!p_re_lu_1_readvariableop_resource*#
_output_shapes
:?*
dtype02
p_re_lu_1/ReadVariableOpu
p_re_lu_1/NegNeg p_re_lu_1/ReadVariableOp:value:0*
T0*#
_output_shapes
:?2
p_re_lu_1/Neg
p_re_lu_1/Neg_1Negconv2d_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
p_re_lu_1/Neg_1|
p_re_lu_1/Relu_1Relup_re_lu_1/Neg_1:y:0*
T0*0
_output_shapes
:??????????2
p_re_lu_1/Relu_1?
p_re_lu_1/mulMulp_re_lu_1/Neg:y:0p_re_lu_1/Relu_1:activations:0*
T0*0
_output_shapes
:??????????2
p_re_lu_1/mul?
p_re_lu_1/addAddV2p_re_lu_1/Relu:activations:0p_re_lu_1/mul:z:0*
T0*0
_output_shapes
:??????????2
p_re_lu_1/add?
conv2d_2/Conv2D/ReadVariableOpReadVariableOp'conv2d_2_conv2d_readvariableop_resource*(
_output_shapes
:??*
dtype02 
conv2d_2/Conv2D/ReadVariableOp?
conv2d_2/Conv2DConv2Dp_re_lu_1/add:z:0&conv2d_2/Conv2D/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2
conv2d_2/Conv2D?
conv2d_2/BiasAdd/ReadVariableOpReadVariableOp(conv2d_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02!
conv2d_2/BiasAdd/ReadVariableOp?
conv2d_2/BiasAddBiasAddconv2d_2/Conv2D:output:0'conv2d_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_2/BiasAdd~
p_re_lu_2/ReluReluconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
p_re_lu_2/Relu?
p_re_lu_2/ReadVariableOpReadVariableOp!p_re_lu_2_readvariableop_resource*#
_output_shapes
:?*
dtype02
p_re_lu_2/ReadVariableOpu
p_re_lu_2/NegNeg p_re_lu_2/ReadVariableOp:value:0*
T0*#
_output_shapes
:?2
p_re_lu_2/Neg
p_re_lu_2/Neg_1Negconv2d_2/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
p_re_lu_2/Neg_1|
p_re_lu_2/Relu_1Relup_re_lu_2/Neg_1:y:0*
T0*0
_output_shapes
:??????????2
p_re_lu_2/Relu_1?
p_re_lu_2/mulMulp_re_lu_2/Neg:y:0p_re_lu_2/Relu_1:activations:0*
T0*0
_output_shapes
:??????????2
p_re_lu_2/mul?
p_re_lu_2/addAddV2p_re_lu_2/Relu:activations:0p_re_lu_2/mul:z:0*
T0*0
_output_shapes
:??????????2
p_re_lu_2/addq
conv2d_transpose/ShapeShapep_re_lu_2/add:z:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicev
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/1v
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/2w
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0p_re_lu_2/add:z:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_transpose/BiasAdd?
p_re_lu_3/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
p_re_lu_3/Relu?
p_re_lu_3/ReadVariableOpReadVariableOp!p_re_lu_3_readvariableop_resource*#
_output_shapes
:?*
dtype02
p_re_lu_3/ReadVariableOpu
p_re_lu_3/NegNeg p_re_lu_3/ReadVariableOp:value:0*
T0*#
_output_shapes
:?2
p_re_lu_3/Neg?
p_re_lu_3/Neg_1Neg!conv2d_transpose/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
p_re_lu_3/Neg_1|
p_re_lu_3/Relu_1Relup_re_lu_3/Neg_1:y:0*
T0*0
_output_shapes
:??????????2
p_re_lu_3/Relu_1?
p_re_lu_3/mulMulp_re_lu_3/Neg:y:0p_re_lu_3/Relu_1:activations:0*
T0*0
_output_shapes
:??????????2
p_re_lu_3/mul?
p_re_lu_3/addAddV2p_re_lu_3/Relu:activations:0p_re_lu_3/mul:z:0*
T0*0
_output_shapes
:??????????2
p_re_lu_3/add|
add/addAddV2p_re_lu_3/add:z:0p_re_lu_1/add:z:0*
T0*0
_output_shapes
:??????????2	
add/addo
conv2d_transpose_1/ShapeShapeadd/add:z:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape?
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack?
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1?
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slicez
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/1z
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/2{
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_1/stack/3?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack?
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stack?
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1?
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0add/add:z:0*
T0*0
_output_shapes
:??????????*
paddingVALID*
strides
2%
#conv2d_transpose_1/conv2d_transpose?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOp?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:??????????2
conv2d_transpose_1/BiasAdd?
p_re_lu_4/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
p_re_lu_4/Relu?
p_re_lu_4/ReadVariableOpReadVariableOp!p_re_lu_4_readvariableop_resource*#
_output_shapes
:?*
dtype02
p_re_lu_4/ReadVariableOpu
p_re_lu_4/NegNeg p_re_lu_4/ReadVariableOp:value:0*
T0*#
_output_shapes
:?2
p_re_lu_4/Neg?
p_re_lu_4/Neg_1Neg#conv2d_transpose_1/BiasAdd:output:0*
T0*0
_output_shapes
:??????????2
p_re_lu_4/Neg_1|
p_re_lu_4/Relu_1Relup_re_lu_4/Neg_1:y:0*
T0*0
_output_shapes
:??????????2
p_re_lu_4/Relu_1?
p_re_lu_4/mulMulp_re_lu_4/Neg:y:0p_re_lu_4/Relu_1:activations:0*
T0*0
_output_shapes
:??????????2
p_re_lu_4/mul?
p_re_lu_4/addAddV2p_re_lu_4/Relu:activations:0p_re_lu_4/mul:z:0*
T0*0
_output_shapes
:??????????2
p_re_lu_4/addu
conv2d_transpose_2/ShapeShapep_re_lu_4/add:z:0*
T0*
_output_shapes
:2
conv2d_transpose_2/Shape?
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_2/strided_slice/stack?
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_1?
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_2?
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_2/strided_slicez
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_2/stack/1z
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :?2
conv2d_transpose_2/stack/2{
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value
B :?2
conv2d_transpose_2/stack/3?
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_2/stack?
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_2/strided_slice_1/stack?
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_1?
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_2?
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_1?
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*(
_output_shapes
:??*
dtype024
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0p_re_lu_4/add:z:0*
T0*0
_output_shapes
:?????????@??*
paddingVALID*
strides
2%
#conv2d_transpose_2/conv2d_transpose?
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02+
)conv2d_transpose_2/BiasAdd/ReadVariableOp?
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*0
_output_shapes
:?????????@??2
conv2d_transpose_2/BiasAdd?
p_re_lu_5/ReluRelu#conv2d_transpose_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@??2
p_re_lu_5/Relu?
p_re_lu_5/ReadVariableOpReadVariableOp!p_re_lu_5_readvariableop_resource*#
_output_shapes
:@??*
dtype02
p_re_lu_5/ReadVariableOpu
p_re_lu_5/NegNeg p_re_lu_5/ReadVariableOp:value:0*
T0*#
_output_shapes
:@??2
p_re_lu_5/Neg?
p_re_lu_5/Neg_1Neg#conv2d_transpose_2/BiasAdd:output:0*
T0*0
_output_shapes
:?????????@??2
p_re_lu_5/Neg_1|
p_re_lu_5/Relu_1Relup_re_lu_5/Neg_1:y:0*
T0*0
_output_shapes
:?????????@??2
p_re_lu_5/Relu_1?
p_re_lu_5/mulMulp_re_lu_5/Neg:y:0p_re_lu_5/Relu_1:activations:0*
T0*0
_output_shapes
:?????????@??2
p_re_lu_5/mul?
p_re_lu_5/addAddV2p_re_lu_5/Relu:activations:0p_re_lu_5/mul:z:0*
T0*0
_output_shapes
:?????????@??2
p_re_lu_5/addu
conv2d_transpose_3/ShapeShapep_re_lu_5/add:z:0*
T0*
_output_shapes
:2
conv2d_transpose_3/Shape?
&conv2d_transpose_3/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_3/strided_slice/stack?
(conv2d_transpose_3/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_1?
(conv2d_transpose_3/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_3/strided_slice/stack_2?
 conv2d_transpose_3/strided_sliceStridedSlice!conv2d_transpose_3/Shape:output:0/conv2d_transpose_3/strided_slice/stack:output:01conv2d_transpose_3/strided_slice/stack_1:output:01conv2d_transpose_3/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_3/strided_slicez
conv2d_transpose_3/stack/1Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose_3/stack/1z
conv2d_transpose_3/stack/2Const*
_output_shapes
: *
dtype0*
value	B :?2
conv2d_transpose_3/stack/2z
conv2d_transpose_3/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_3/stack/3?
conv2d_transpose_3/stackPack)conv2d_transpose_3/strided_slice:output:0#conv2d_transpose_3/stack/1:output:0#conv2d_transpose_3/stack/2:output:0#conv2d_transpose_3/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_3/stack?
(conv2d_transpose_3/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_3/strided_slice_1/stack?
*conv2d_transpose_3/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_1?
*conv2d_transpose_3/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_3/strided_slice_1/stack_2?
"conv2d_transpose_3/strided_slice_1StridedSlice!conv2d_transpose_3/stack:output:01conv2d_transpose_3/strided_slice_1/stack:output:03conv2d_transpose_3/strided_slice_1/stack_1:output:03conv2d_transpose_3/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_3/strided_slice_1?
2conv2d_transpose_3/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_3_conv2d_transpose_readvariableop_resource*'
_output_shapes
:?*
dtype024
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_3/conv2d_transposeConv2DBackpropInput!conv2d_transpose_3/stack:output:0:conv2d_transpose_3/conv2d_transpose/ReadVariableOp:value:0p_re_lu_5/add:z:0*
T0*/
_output_shapes
:?????????@?*
paddingSAME*
strides
2%
#conv2d_transpose_3/conv2d_transpose?
)conv2d_transpose_3/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_3_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_3/BiasAdd/ReadVariableOp?
conv2d_transpose_3/BiasAddBiasAdd,conv2d_transpose_3/conv2d_transpose:output:01conv2d_transpose_3/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@?2
conv2d_transpose_3/BiasAdd?
IdentityIdentity#conv2d_transpose_3/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp ^conv2d_2/BiasAdd/ReadVariableOp^conv2d_2/Conv2D/ReadVariableOp(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp*^conv2d_transpose_3/BiasAdd/ReadVariableOp3^conv2d_transpose_3/conv2d_transpose/ReadVariableOp^p_re_lu/ReadVariableOp^p_re_lu_1/ReadVariableOp^p_re_lu_2/ReadVariableOp^p_re_lu_3/ReadVariableOp^p_re_lu_4/ReadVariableOp^p_re_lu_5/ReadVariableOp*
T0*/
_output_shapes
:?????????@?2

Identity"
identityIdentity:output:0*~
_input_shapesm
k:?????????@?::::::::::::::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2B
conv2d_2/BiasAdd/ReadVariableOpconv2d_2/BiasAdd/ReadVariableOp2@
conv2d_2/Conv2D/ReadVariableOpconv2d_2/Conv2D/ReadVariableOp2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_3/BiasAdd/ReadVariableOp)conv2d_transpose_3/BiasAdd/ReadVariableOp2h
2conv2d_transpose_3/conv2d_transpose/ReadVariableOp2conv2d_transpose_3/conv2d_transpose/ReadVariableOp20
p_re_lu/ReadVariableOpp_re_lu/ReadVariableOp24
p_re_lu_1/ReadVariableOpp_re_lu_1/ReadVariableOp24
p_re_lu_2/ReadVariableOpp_re_lu_2/ReadVariableOp24
p_re_lu_3/ReadVariableOpp_re_lu_3/ReadVariableOp24
p_re_lu_4/ReadVariableOpp_re_lu_4/ReadVariableOp24
p_re_lu_5/ReadVariableOpp_re_lu_5/ReadVariableOp:W S
/
_output_shapes
:?????????@?
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????@?N
conv2d_transpose_38
StatefulPartitionedCall:0?????????@?tensorflow/serving/predict:??
??
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
layer_with_weights-2
layer-3
layer_with_weights-3
layer-4
layer_with_weights-4
layer-5
layer_with_weights-5
layer-6
layer_with_weights-6
layer-7
	layer_with_weights-7
	layer-8

layer-9
layer_with_weights-8
layer-10
layer_with_weights-9
layer-11
layer_with_weights-10
layer-12
layer_with_weights-11
layer-13
layer_with_weights-12
layer-14
trainable_variables
	variables
regularization_losses
	keras_api

signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature"?}
_tf_keras_network?}{"class_name": "Functional", "name": "model", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 63, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "p_re_lu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "name": "p_re_lu", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["p_re_lu", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "p_re_lu_1", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "name": "p_re_lu_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["p_re_lu_1", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "p_re_lu_2", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "name": "p_re_lu_2", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose", "inbound_nodes": [[["p_re_lu_2", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "p_re_lu_3", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "name": "p_re_lu_3", "inbound_nodes": [[["conv2d_transpose", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["p_re_lu_3", 0, 0, {}], ["p_re_lu_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_1", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "p_re_lu_4", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "name": "p_re_lu_4", "inbound_nodes": [[["conv2d_transpose_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_2", "inbound_nodes": [[["p_re_lu_4", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "p_re_lu_5", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "name": "p_re_lu_5", "inbound_nodes": [[["conv2d_transpose_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_3", "inbound_nodes": [[["p_re_lu_5", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_transpose_3", 0, 0]]}, "input_spec": [{"class_name": "InputSpec", "config": {"dtype": null, "shape": {"class_name": "__tuple__", "items": [null, 64, 63, 1]}, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}], "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 63, 1]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "model", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 63, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Conv2D", "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "p_re_lu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "name": "p_re_lu", "inbound_nodes": [[["conv2d", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_1", "inbound_nodes": [[["p_re_lu", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "p_re_lu_1", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "name": "p_re_lu_1", "inbound_nodes": [[["conv2d_1", 0, 0, {}]]]}, {"class_name": "Conv2D", "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "name": "conv2d_2", "inbound_nodes": [[["p_re_lu_1", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "p_re_lu_2", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "name": "p_re_lu_2", "inbound_nodes": [[["conv2d_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose", "inbound_nodes": [[["p_re_lu_2", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "p_re_lu_3", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "name": "p_re_lu_3", "inbound_nodes": [[["conv2d_transpose", 0, 0, {}]]]}, {"class_name": "Add", "config": {"name": "add", "trainable": true, "dtype": "float32"}, "name": "add", "inbound_nodes": [[["p_re_lu_3", 0, 0, {}], ["p_re_lu_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_1", "inbound_nodes": [[["add", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "p_re_lu_4", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "name": "p_re_lu_4", "inbound_nodes": [[["conv2d_transpose_1", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_2", "inbound_nodes": [[["p_re_lu_4", 0, 0, {}]]]}, {"class_name": "PReLU", "config": {"name": "p_re_lu_5", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "name": "p_re_lu_5", "inbound_nodes": [[["conv2d_transpose_2", 0, 0, {}]]]}, {"class_name": "Conv2DTranspose", "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "name": "conv2d_transpose_3", "inbound_nodes": [[["p_re_lu_5", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["conv2d_transpose_3", 0, 0]]}}}
?"?
_tf_keras_input_layer?{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 63, 1]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 64, 63, 1]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
?	

kernel
bias
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 63, 1]}}
?
	alpha
regularization_losses
trainable_variables
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "PReLU", "name": "p_re_lu", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "p_re_lu", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 31, 31, 128]}}
?	

 kernel
!bias
"regularization_losses
#trainable_variables
$	variables
%	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 31, 31, 128]}}
?
	&alpha
'regularization_losses
(trainable_variables
)	variables
*	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "PReLU", "name": "p_re_lu_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "p_re_lu_1", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 15, 256]}}
?	

+kernel
,bias
-regularization_losses
.trainable_variables
/	variables
0	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_2", "trainable": true, "dtype": "float32", "filters": 512, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 15, 256]}}
?
	1alpha
2regularization_losses
3trainable_variables
4	variables
5	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "PReLU", "name": "p_re_lu_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "p_re_lu_2", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 512]}}
?


6kernel
7bias
8regularization_losses
9trainable_variables
:	variables
;	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 256, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 512}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 512]}}
?
	<alpha
=regularization_losses
>trainable_variables
?	variables
@	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "PReLU", "name": "p_re_lu_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "p_re_lu_3", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 15, 256]}}
?
Aregularization_losses
Btrainable_variables
C	variables
D	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Add", "name": "add", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "add", "trainable": true, "dtype": "float32"}, "build_input_shape": [{"class_name": "TensorShape", "items": [null, 15, 15, 256]}, {"class_name": "TensorShape", "items": [null, 15, 15, 256]}]}
?


Ekernel
Fbias
Gregularization_losses
Htrainable_variables
I	variables
J	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 256}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 15, 15, 256]}}
?
	Kalpha
Lregularization_losses
Mtrainable_variables
N	variables
O	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "PReLU", "name": "p_re_lu_4", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "p_re_lu_4", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 31, 31, 128]}}
?


Pkernel
Qbias
Rregularization_losses
Strainable_variables
T	variables
U	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 128, "kernel_size": {"class_name": "__tuple__", "items": [4, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 31, 31, 128]}}
?
	Valpha
Wregularization_losses
Xtrainable_variables
Y	variables
Z	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "PReLU", "name": "p_re_lu_5", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "p_re_lu_5", "trainable": true, "dtype": "float32", "alpha_initializer": {"class_name": "Zeros", "config": {}}, "alpha_regularizer": null, "alpha_constraint": null, "shared_axes": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 63, 128]}}
?


[kernel
\bias
]regularization_losses
^trainable_variables
_	variables
`	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_3", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_3", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 128}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 64, 63, 128]}}
?
0
1
2
 3
!4
&5
+6
,7
18
69
710
<11
E12
F13
K14
P15
Q16
V17
[18
\19"
trackable_list_wrapper
?
0
1
2
 3
!4
&5
+6
,7
18
69
710
<11
E12
F13
K14
P15
Q16
V17
[18
\19"
trackable_list_wrapper
 "
trackable_list_wrapper
?

alayers
bnon_trainable_variables
clayer_regularization_losses
dmetrics
elayer_metrics
trainable_variables
	variables
regularization_losses
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
(:&?2conv2d/kernel
:?2conv2d/bias
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
?
fnon_trainable_variables
regularization_losses
glayer_regularization_losses
hlayer_metrics
imetrics
trainable_variables
	variables

jlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
$:"?2p_re_lu/alpha
 "
trackable_list_wrapper
'
0"
trackable_list_wrapper
'
0"
trackable_list_wrapper
?
knon_trainable_variables
regularization_losses
llayer_regularization_losses
mlayer_metrics
nmetrics
trainable_variables
	variables

olayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)??2conv2d_1/kernel
:?2conv2d_1/bias
 "
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
.
 0
!1"
trackable_list_wrapper
?
pnon_trainable_variables
"regularization_losses
qlayer_regularization_losses
rlayer_metrics
smetrics
#trainable_variables
$	variables

tlayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$?2p_re_lu_1/alpha
 "
trackable_list_wrapper
'
&0"
trackable_list_wrapper
'
&0"
trackable_list_wrapper
?
unon_trainable_variables
'regularization_losses
vlayer_regularization_losses
wlayer_metrics
xmetrics
(trainable_variables
)	variables

ylayers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
+:)??2conv2d_2/kernel
:?2conv2d_2/bias
 "
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
.
+0
,1"
trackable_list_wrapper
?
znon_trainable_variables
-regularization_losses
{layer_regularization_losses
|layer_metrics
}metrics
.trainable_variables
/	variables

~layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$?2p_re_lu_2/alpha
 "
trackable_list_wrapper
'
10"
trackable_list_wrapper
'
10"
trackable_list_wrapper
?
non_trainable_variables
2regularization_losses
 ?layer_regularization_losses
?layer_metrics
?metrics
3trainable_variables
4	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
3:1??2conv2d_transpose/kernel
$:"?2conv2d_transpose/bias
 "
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
.
60
71"
trackable_list_wrapper
?
?non_trainable_variables
8regularization_losses
 ?layer_regularization_losses
?layer_metrics
?metrics
9trainable_variables
:	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$?2p_re_lu_3/alpha
 "
trackable_list_wrapper
'
<0"
trackable_list_wrapper
'
<0"
trackable_list_wrapper
?
?non_trainable_variables
=regularization_losses
 ?layer_regularization_losses
?layer_metrics
?metrics
>trainable_variables
?	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?non_trainable_variables
Aregularization_losses
 ?layer_regularization_losses
?layer_metrics
?metrics
Btrainable_variables
C	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
5:3??2conv2d_transpose_1/kernel
&:$?2conv2d_transpose_1/bias
 "
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
.
E0
F1"
trackable_list_wrapper
?
?non_trainable_variables
Gregularization_losses
 ?layer_regularization_losses
?layer_metrics
?metrics
Htrainable_variables
I	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$?2p_re_lu_4/alpha
 "
trackable_list_wrapper
'
K0"
trackable_list_wrapper
'
K0"
trackable_list_wrapper
?
?non_trainable_variables
Lregularization_losses
 ?layer_regularization_losses
?layer_metrics
?metrics
Mtrainable_variables
N	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
5:3??2conv2d_transpose_2/kernel
&:$?2conv2d_transpose_2/bias
 "
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
.
P0
Q1"
trackable_list_wrapper
?
?non_trainable_variables
Rregularization_losses
 ?layer_regularization_losses
?layer_metrics
?metrics
Strainable_variables
T	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
&:$@??2p_re_lu_5/alpha
 "
trackable_list_wrapper
'
V0"
trackable_list_wrapper
'
V0"
trackable_list_wrapper
?
?non_trainable_variables
Wregularization_losses
 ?layer_regularization_losses
?layer_metrics
?metrics
Xtrainable_variables
Y	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
4:2?2conv2d_transpose_3/kernel
%:#2conv2d_transpose_3/bias
 "
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
.
[0
\1"
trackable_list_wrapper
?
?non_trainable_variables
]regularization_losses
 ?layer_regularization_losses
?layer_metrics
?metrics
^trainable_variables
_	variables
?layers
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?
0
1
2
3
4
5
6
7
	8

9
10
11
12
13
14"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?2?
(__inference_model_layer_call_fn_45385847
(__inference_model_layer_call_fn_45385461
(__inference_model_layer_call_fn_45385892
(__inference_model_layer_call_fn_45385358?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
C__inference_model_layer_call_and_return_conditional_losses_45385655
C__inference_model_layer_call_and_return_conditional_losses_45385802
C__inference_model_layer_call_and_return_conditional_losses_45385254
C__inference_model_layer_call_and_return_conditional_losses_45385196?
???
FullArgSpec1
args)?&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults?
p 

 

kwonlyargs? 
kwonlydefaults? 
annotations? *
 
?2?
#__inference__wrapped_model_45384746?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????@?
?2?
)__inference_conv2d_layer_call_fn_45385911?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
D__inference_conv2d_layer_call_and_return_conditional_losses_45385902?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
*__inference_p_re_lu_layer_call_fn_45384767?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
E__inference_p_re_lu_layer_call_and_return_conditional_losses_45384759?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
+__inference_conv2d_1_layer_call_fn_45385930?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_1_layer_call_and_return_conditional_losses_45385921?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_p_re_lu_1_layer_call_fn_45384788?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_p_re_lu_1_layer_call_and_return_conditional_losses_45384780?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
+__inference_conv2d_2_layer_call_fn_45385949?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
F__inference_conv2d_2_layer_call_and_return_conditional_losses_45385940?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
,__inference_p_re_lu_2_layer_call_fn_45384809?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_p_re_lu_2_layer_call_and_return_conditional_losses_45384801?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
3__inference_conv2d_transpose_layer_call_fn_45384857?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
N__inference_conv2d_transpose_layer_call_and_return_conditional_losses_45384847?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
,__inference_p_re_lu_3_layer_call_fn_45384878?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_p_re_lu_3_layer_call_and_return_conditional_losses_45384870?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
&__inference_add_layer_call_fn_45385961?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
A__inference_add_layer_call_and_return_conditional_losses_45385955?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
5__inference_conv2d_transpose_1_layer_call_fn_45384926?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
P__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_45384916?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
,__inference_p_re_lu_4_layer_call_fn_45384947?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_p_re_lu_4_layer_call_and_return_conditional_losses_45384939?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
5__inference_conv2d_transpose_2_layer_call_fn_45384995?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
P__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_45384985?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
,__inference_p_re_lu_5_layer_call_fn_45385016?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
G__inference_p_re_lu_5_layer_call_and_return_conditional_losses_45385008?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *@?=
;?84????????????????????????????????????
?2?
5__inference_conv2d_transpose_3_layer_call_fn_45385060?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?2?
P__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_45385050?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *8?5
3?0,????????????????????????????
?B?
&__inference_signature_wrapper_45385508input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 ?
#__inference__wrapped_model_45384746? !&+,167<EFKPQV[\8?5
.?+
)?&
input_1?????????@?
? "O?L
J
conv2d_transpose_34?1
conv2d_transpose_3?????????@??
A__inference_add_layer_call_and_return_conditional_losses_45385955?l?i
b?_
]?Z
+?(
inputs/0??????????
+?(
inputs/1??????????
? ".?+
$?!
0??????????
? ?
&__inference_add_layer_call_fn_45385961?l?i
b?_
]?Z
+?(
inputs/0??????????
+?(
inputs/1??????????
? "!????????????
F__inference_conv2d_1_layer_call_and_return_conditional_losses_45385921n !8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
+__inference_conv2d_1_layer_call_fn_45385930a !8?5
.?+
)?&
inputs??????????
? "!????????????
F__inference_conv2d_2_layer_call_and_return_conditional_losses_45385940n+,8?5
.?+
)?&
inputs??????????
? ".?+
$?!
0??????????
? ?
+__inference_conv2d_2_layer_call_fn_45385949a+,8?5
.?+
)?&
inputs??????????
? "!????????????
D__inference_conv2d_layer_call_and_return_conditional_losses_45385902m7?4
-?*
(?%
inputs?????????@?
? ".?+
$?!
0??????????
? ?
)__inference_conv2d_layer_call_fn_45385911`7?4
-?*
(?%
inputs?????????@?
? "!????????????
P__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_45384916?EFJ?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
5__inference_conv2d_transpose_1_layer_call_fn_45384926?EFJ?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
P__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_45384985?PQJ?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
5__inference_conv2d_transpose_2_layer_call_fn_45384995?PQJ?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
P__inference_conv2d_transpose_3_layer_call_and_return_conditional_losses_45385050?[\J?G
@?=
;?8
inputs,????????????????????????????
? "??<
5?2
0+???????????????????????????
? ?
5__inference_conv2d_transpose_3_layer_call_fn_45385060?[\J?G
@?=
;?8
inputs,????????????????????????????
? "2?/+????????????????????????????
N__inference_conv2d_transpose_layer_call_and_return_conditional_losses_45384847?67J?G
@?=
;?8
inputs,????????????????????????????
? "@?=
6?3
0,????????????????????????????
? ?
3__inference_conv2d_transpose_layer_call_fn_45384857?67J?G
@?=
;?8
inputs,????????????????????????????
? "3?0,?????????????????????????????
C__inference_model_layer_call_and_return_conditional_losses_45385196? !&+,167<EFKPQV[\@?=
6?3
)?&
input_1?????????@?
p

 
? "??<
5?2
0+???????????????????????????
? ?
C__inference_model_layer_call_and_return_conditional_losses_45385254? !&+,167<EFKPQV[\@?=
6?3
)?&
input_1?????????@?
p 

 
? "??<
5?2
0+???????????????????????????
? ?
C__inference_model_layer_call_and_return_conditional_losses_45385655? !&+,167<EFKPQV[\??<
5?2
(?%
inputs?????????@?
p

 
? "-?*
#? 
0?????????@?
? ?
C__inference_model_layer_call_and_return_conditional_losses_45385802? !&+,167<EFKPQV[\??<
5?2
(?%
inputs?????????@?
p 

 
? "-?*
#? 
0?????????@?
? ?
(__inference_model_layer_call_fn_45385358? !&+,167<EFKPQV[\@?=
6?3
)?&
input_1?????????@?
p

 
? "2?/+????????????????????????????
(__inference_model_layer_call_fn_45385461? !&+,167<EFKPQV[\@?=
6?3
)?&
input_1?????????@?
p 

 
? "2?/+????????????????????????????
(__inference_model_layer_call_fn_45385847? !&+,167<EFKPQV[\??<
5?2
(?%
inputs?????????@?
p

 
? "2?/+????????????????????????????
(__inference_model_layer_call_fn_45385892? !&+,167<EFKPQV[\??<
5?2
(?%
inputs?????????@?
p 

 
? "2?/+????????????????????????????
G__inference_p_re_lu_1_layer_call_and_return_conditional_losses_45384780?&R?O
H?E
C?@
inputs4????????????????????????????????????
? ".?+
$?!
0??????????
? ?
,__inference_p_re_lu_1_layer_call_fn_45384788z&R?O
H?E
C?@
inputs4????????????????????????????????????
? "!????????????
G__inference_p_re_lu_2_layer_call_and_return_conditional_losses_45384801?1R?O
H?E
C?@
inputs4????????????????????????????????????
? ".?+
$?!
0??????????
? ?
,__inference_p_re_lu_2_layer_call_fn_45384809z1R?O
H?E
C?@
inputs4????????????????????????????????????
? "!????????????
G__inference_p_re_lu_3_layer_call_and_return_conditional_losses_45384870?<R?O
H?E
C?@
inputs4????????????????????????????????????
? ".?+
$?!
0??????????
? ?
,__inference_p_re_lu_3_layer_call_fn_45384878z<R?O
H?E
C?@
inputs4????????????????????????????????????
? "!????????????
G__inference_p_re_lu_4_layer_call_and_return_conditional_losses_45384939?KR?O
H?E
C?@
inputs4????????????????????????????????????
? ".?+
$?!
0??????????
? ?
,__inference_p_re_lu_4_layer_call_fn_45384947zKR?O
H?E
C?@
inputs4????????????????????????????????????
? "!????????????
G__inference_p_re_lu_5_layer_call_and_return_conditional_losses_45385008?VR?O
H?E
C?@
inputs4????????????????????????????????????
? ".?+
$?!
0?????????@??
? ?
,__inference_p_re_lu_5_layer_call_fn_45385016zVR?O
H?E
C?@
inputs4????????????????????????????????????
? "!??????????@???
E__inference_p_re_lu_layer_call_and_return_conditional_losses_45384759?R?O
H?E
C?@
inputs4????????????????????????????????????
? ".?+
$?!
0??????????
? ?
*__inference_p_re_lu_layer_call_fn_45384767zR?O
H?E
C?@
inputs4????????????????????????????????????
? "!????????????
&__inference_signature_wrapper_45385508? !&+,167<EFKPQV[\C?@
? 
9?6
4
input_1)?&
input_1?????????@?"O?L
J
conv2d_transpose_34?1
conv2d_transpose_3?????????@?