Ŕ
ß¸
:
Add
x"T
y"T
z"T"
Ttype:
2	

ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
ś
AsString

input"T

output"
Ttype:
	2	
"
	precisionint˙˙˙˙˙˙˙˙˙"

scientificbool( "
shortestbool( "
widthint˙˙˙˙˙˙˙˙˙"
fillstring 
x
Assign
ref"T

value"T

output_ref"T"	
Ttype"
validate_shapebool("
use_lockingbool(
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
ë
Conv2D

input"T
filter"T
output"T"
Ttype:
2"
strides	list(int)"
use_cudnn_on_gpubool(""
paddingstring:
SAMEVALID"-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

.
Identity

input"T
output"T"	
Ttype
p
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:
	2
Ô
MaxPool

input"T
output"T"
Ttype0:
2	"
ksize	list(int)(0"
strides	list(int)(0""
paddingstring:
SAMEVALID":
data_formatstringNHWC:
NHWCNCHWNCHW_VECT_C
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(
=
Mul
x"T
y"T
z"T"
Ttype:
2	
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
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
D
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0
H
ShardedFilename
basename	
shard

num_shards
filename
9
Softmax
logits"T
softmax"T"
Ttype:
2
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
:
Sub
x"T
y"T
z"T"
Ttype:
2	
s

VariableV2
ref"dtype"
shapeshape"
dtypetype"
	containerstring "
shared_namestring "serve*1.5.02v1.5.0-0-g37aa430d84°ő

global_step/Initializer/zerosConst*
dtype0	*
_class
loc:@global_step*
value	B	 R *
_output_shapes
: 

global_step
VariableV2*
	container *
_output_shapes
: *
dtype0	*
shape: *
_class
loc:@global_step*
shared_name 
˛
global_step/AssignAssignglobal_stepglobal_step/Initializer/zeros*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0	*
_output_shapes
: 
j
global_step/readIdentityglobal_step*
_class
loc:@global_step*
T0	*
_output_shapes
: 
p
PlaceholderPlaceholder*
dtype0*
shape:˙˙˙˙˙˙˙˙˙*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
f
Reshape/shapeConst*
dtype0*%
valueB"˙˙˙˙         *
_output_shapes
:
v
ReshapeReshapePlaceholderReshape/shape*/
_output_shapes
:˙˙˙˙˙˙˙˙˙*
Tshape0*
T0
Š
.conv2d/kernel/Initializer/random_uniform/shapeConst*
dtype0* 
_class
loc:@conv2d/kernel*%
valueB"             *
_output_shapes
:

,conv2d/kernel/Initializer/random_uniform/minConst*
dtype0* 
_class
loc:@conv2d/kernel*
valueB
 *n§Ž˝*
_output_shapes
: 

,conv2d/kernel/Initializer/random_uniform/maxConst*
dtype0* 
_class
loc:@conv2d/kernel*
valueB
 *n§Ž=*
_output_shapes
: 
đ
6conv2d/kernel/Initializer/random_uniform/RandomUniformRandomUniform.conv2d/kernel/Initializer/random_uniform/shape*&
_output_shapes
: *
dtype0*
seed2 *

seed *
T0* 
_class
loc:@conv2d/kernel
Ň
,conv2d/kernel/Initializer/random_uniform/subSub,conv2d/kernel/Initializer/random_uniform/max,conv2d/kernel/Initializer/random_uniform/min* 
_class
loc:@conv2d/kernel*
T0*
_output_shapes
: 
ě
,conv2d/kernel/Initializer/random_uniform/mulMul6conv2d/kernel/Initializer/random_uniform/RandomUniform,conv2d/kernel/Initializer/random_uniform/sub* 
_class
loc:@conv2d/kernel*
T0*&
_output_shapes
: 
Ţ
(conv2d/kernel/Initializer/random_uniformAdd,conv2d/kernel/Initializer/random_uniform/mul,conv2d/kernel/Initializer/random_uniform/min* 
_class
loc:@conv2d/kernel*
T0*&
_output_shapes
: 
ł
conv2d/kernel
VariableV2*
	container *&
_output_shapes
: *
dtype0*
shape: * 
_class
loc:@conv2d/kernel*
shared_name 
Ó
conv2d/kernel/AssignAssignconv2d/kernel(conv2d/kernel/Initializer/random_uniform*
validate_shape(* 
_class
loc:@conv2d/kernel*
use_locking(*
T0*&
_output_shapes
: 

conv2d/kernel/readIdentityconv2d/kernel* 
_class
loc:@conv2d/kernel*
T0*&
_output_shapes
: 

conv2d/bias/Initializer/zerosConst*
dtype0*
_class
loc:@conv2d/bias*
valueB *    *
_output_shapes
: 

conv2d/bias
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: *
_class
loc:@conv2d/bias*
shared_name 
ś
conv2d/bias/AssignAssignconv2d/biasconv2d/bias/Initializer/zeros*
validate_shape(*
_class
loc:@conv2d/bias*
use_locking(*
T0*
_output_shapes
: 
n
conv2d/bias/readIdentityconv2d/bias*
_class
loc:@conv2d/bias*
T0*
_output_shapes
: 
e
conv2d/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
Ü
conv2d/Conv2DConv2DReshapeconv2d/kernel/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0*
	dilations


conv2d/BiasAddBiasAddconv2d/Conv2Dconv2d/bias/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
data_formatNHWC*
T0
]
conv2d/ReluReluconv2d/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
­
0conv2d_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*"
_class
loc:@conv2d_1/kernel*%
valueB"             *
_output_shapes
:

.conv2d_1/kernel/Initializer/random_uniform/minConst*
dtype0*"
_class
loc:@conv2d_1/kernel*
valueB
 *n§Ž˝*
_output_shapes
: 

.conv2d_1/kernel/Initializer/random_uniform/maxConst*
dtype0*"
_class
loc:@conv2d_1/kernel*
valueB
 *n§Ž=*
_output_shapes
: 
ö
8conv2d_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_1/kernel/Initializer/random_uniform/shape*&
_output_shapes
: *
dtype0*
seed2 *

seed *
T0*"
_class
loc:@conv2d_1/kernel
Ú
.conv2d_1/kernel/Initializer/random_uniform/subSub.conv2d_1/kernel/Initializer/random_uniform/max.conv2d_1/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_1/kernel*
T0*
_output_shapes
: 
ô
.conv2d_1/kernel/Initializer/random_uniform/mulMul8conv2d_1/kernel/Initializer/random_uniform/RandomUniform.conv2d_1/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_1/kernel*
T0*&
_output_shapes
: 
ć
*conv2d_1/kernel/Initializer/random_uniformAdd.conv2d_1/kernel/Initializer/random_uniform/mul.conv2d_1/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_1/kernel*
T0*&
_output_shapes
: 
ˇ
conv2d_1/kernel
VariableV2*
	container *&
_output_shapes
: *
dtype0*
shape: *"
_class
loc:@conv2d_1/kernel*
shared_name 
Ű
conv2d_1/kernel/AssignAssignconv2d_1/kernel*conv2d_1/kernel/Initializer/random_uniform*
validate_shape(*"
_class
loc:@conv2d_1/kernel*
use_locking(*
T0*&
_output_shapes
: 

conv2d_1/kernel/readIdentityconv2d_1/kernel*"
_class
loc:@conv2d_1/kernel*
T0*&
_output_shapes
: 

conv2d_1/bias/Initializer/zerosConst*
dtype0* 
_class
loc:@conv2d_1/bias*
valueB *    *
_output_shapes
: 

conv2d_1/bias
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: * 
_class
loc:@conv2d_1/bias*
shared_name 
ž
conv2d_1/bias/AssignAssignconv2d_1/biasconv2d_1/bias/Initializer/zeros*
validate_shape(* 
_class
loc:@conv2d_1/bias*
use_locking(*
T0*
_output_shapes
: 
t
conv2d_1/bias/readIdentityconv2d_1/bias* 
_class
loc:@conv2d_1/bias*
T0*
_output_shapes
: 
g
conv2d_2/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
ŕ
conv2d_2/Conv2DConv2DReshapeconv2d_1/kernel/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0*
	dilations


conv2d_2/BiasAddBiasAddconv2d_2/Conv2Dconv2d_1/bias/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
data_formatNHWC*
T0
a
conv2d_2/ReluReluconv2d_2/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
­
0conv2d_2/kernel/Initializer/random_uniform/shapeConst*
dtype0*"
_class
loc:@conv2d_2/kernel*%
valueB"             *
_output_shapes
:

.conv2d_2/kernel/Initializer/random_uniform/minConst*
dtype0*"
_class
loc:@conv2d_2/kernel*
valueB
 *n§Ž˝*
_output_shapes
: 

.conv2d_2/kernel/Initializer/random_uniform/maxConst*
dtype0*"
_class
loc:@conv2d_2/kernel*
valueB
 *n§Ž=*
_output_shapes
: 
ö
8conv2d_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_2/kernel/Initializer/random_uniform/shape*&
_output_shapes
: *
dtype0*
seed2 *

seed *
T0*"
_class
loc:@conv2d_2/kernel
Ú
.conv2d_2/kernel/Initializer/random_uniform/subSub.conv2d_2/kernel/Initializer/random_uniform/max.conv2d_2/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_2/kernel*
T0*
_output_shapes
: 
ô
.conv2d_2/kernel/Initializer/random_uniform/mulMul8conv2d_2/kernel/Initializer/random_uniform/RandomUniform.conv2d_2/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_2/kernel*
T0*&
_output_shapes
: 
ć
*conv2d_2/kernel/Initializer/random_uniformAdd.conv2d_2/kernel/Initializer/random_uniform/mul.conv2d_2/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_2/kernel*
T0*&
_output_shapes
: 
ˇ
conv2d_2/kernel
VariableV2*
	container *&
_output_shapes
: *
dtype0*
shape: *"
_class
loc:@conv2d_2/kernel*
shared_name 
Ű
conv2d_2/kernel/AssignAssignconv2d_2/kernel*conv2d_2/kernel/Initializer/random_uniform*
validate_shape(*"
_class
loc:@conv2d_2/kernel*
use_locking(*
T0*&
_output_shapes
: 

conv2d_2/kernel/readIdentityconv2d_2/kernel*"
_class
loc:@conv2d_2/kernel*
T0*&
_output_shapes
: 

conv2d_2/bias/Initializer/zerosConst*
dtype0* 
_class
loc:@conv2d_2/bias*
valueB *    *
_output_shapes
: 

conv2d_2/bias
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: * 
_class
loc:@conv2d_2/bias*
shared_name 
ž
conv2d_2/bias/AssignAssignconv2d_2/biasconv2d_2/bias/Initializer/zeros*
validate_shape(* 
_class
loc:@conv2d_2/bias*
use_locking(*
T0*
_output_shapes
: 
t
conv2d_2/bias/readIdentityconv2d_2/bias* 
_class
loc:@conv2d_2/bias*
T0*
_output_shapes
: 
g
conv2d_3/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
ŕ
conv2d_3/Conv2DConv2DReshapeconv2d_2/kernel/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0*
	dilations


conv2d_3/BiasAddBiasAddconv2d_3/Conv2Dconv2d_2/bias/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
data_formatNHWC*
T0
a
conv2d_3/ReluReluconv2d_3/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
­
0conv2d_3/kernel/Initializer/random_uniform/shapeConst*
dtype0*"
_class
loc:@conv2d_3/kernel*%
valueB"             *
_output_shapes
:

.conv2d_3/kernel/Initializer/random_uniform/minConst*
dtype0*"
_class
loc:@conv2d_3/kernel*
valueB
 *n§Ž˝*
_output_shapes
: 

.conv2d_3/kernel/Initializer/random_uniform/maxConst*
dtype0*"
_class
loc:@conv2d_3/kernel*
valueB
 *n§Ž=*
_output_shapes
: 
ö
8conv2d_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_3/kernel/Initializer/random_uniform/shape*&
_output_shapes
: *
dtype0*
seed2 *

seed *
T0*"
_class
loc:@conv2d_3/kernel
Ú
.conv2d_3/kernel/Initializer/random_uniform/subSub.conv2d_3/kernel/Initializer/random_uniform/max.conv2d_3/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_3/kernel*
T0*
_output_shapes
: 
ô
.conv2d_3/kernel/Initializer/random_uniform/mulMul8conv2d_3/kernel/Initializer/random_uniform/RandomUniform.conv2d_3/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_3/kernel*
T0*&
_output_shapes
: 
ć
*conv2d_3/kernel/Initializer/random_uniformAdd.conv2d_3/kernel/Initializer/random_uniform/mul.conv2d_3/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_3/kernel*
T0*&
_output_shapes
: 
ˇ
conv2d_3/kernel
VariableV2*
	container *&
_output_shapes
: *
dtype0*
shape: *"
_class
loc:@conv2d_3/kernel*
shared_name 
Ű
conv2d_3/kernel/AssignAssignconv2d_3/kernel*conv2d_3/kernel/Initializer/random_uniform*
validate_shape(*"
_class
loc:@conv2d_3/kernel*
use_locking(*
T0*&
_output_shapes
: 

conv2d_3/kernel/readIdentityconv2d_3/kernel*"
_class
loc:@conv2d_3/kernel*
T0*&
_output_shapes
: 

conv2d_3/bias/Initializer/zerosConst*
dtype0* 
_class
loc:@conv2d_3/bias*
valueB *    *
_output_shapes
: 

conv2d_3/bias
VariableV2*
	container *
_output_shapes
: *
dtype0*
shape: * 
_class
loc:@conv2d_3/bias*
shared_name 
ž
conv2d_3/bias/AssignAssignconv2d_3/biasconv2d_3/bias/Initializer/zeros*
validate_shape(* 
_class
loc:@conv2d_3/bias*
use_locking(*
T0*
_output_shapes
: 
t
conv2d_3/bias/readIdentityconv2d_3/bias* 
_class
loc:@conv2d_3/bias*
T0*
_output_shapes
: 
g
conv2d_4/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
ŕ
conv2d_4/Conv2DConv2DReshapeconv2d_3/kernel/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0*
	dilations


conv2d_4/BiasAddBiasAddconv2d_4/Conv2Dconv2d_3/bias/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
data_formatNHWC*
T0
a
conv2d_4/ReluReluconv2d_4/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ 
ş
max_pooling2d/MaxPoolMaxPoolconv2d/Relu*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
data_formatNHWC*
paddingVALID*
strides
*
ksize
*
T0
ž
max_pooling2d_2/MaxPoolMaxPoolconv2d_2/Relu*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
data_formatNHWC*
paddingVALID*
strides
*
ksize
*
T0
ž
max_pooling2d_3/MaxPoolMaxPoolconv2d_3/Relu*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
data_formatNHWC*
paddingVALID*
strides
*
ksize
*
T0
ž
max_pooling2d_4/MaxPoolMaxPoolconv2d_4/Relu*/
_output_shapes
:˙˙˙˙˙˙˙˙˙ *
data_formatNHWC*
paddingVALID*
strides
*
ksize
*
T0
­
0conv2d_4/kernel/Initializer/random_uniform/shapeConst*
dtype0*"
_class
loc:@conv2d_4/kernel*%
valueB"          @   *
_output_shapes
:

.conv2d_4/kernel/Initializer/random_uniform/minConst*
dtype0*"
_class
loc:@conv2d_4/kernel*
valueB
 *ÍĚL˝*
_output_shapes
: 

.conv2d_4/kernel/Initializer/random_uniform/maxConst*
dtype0*"
_class
loc:@conv2d_4/kernel*
valueB
 *ÍĚL=*
_output_shapes
: 
ö
8conv2d_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_4/kernel/Initializer/random_uniform/shape*&
_output_shapes
: @*
dtype0*
seed2 *

seed *
T0*"
_class
loc:@conv2d_4/kernel
Ú
.conv2d_4/kernel/Initializer/random_uniform/subSub.conv2d_4/kernel/Initializer/random_uniform/max.conv2d_4/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_4/kernel*
T0*
_output_shapes
: 
ô
.conv2d_4/kernel/Initializer/random_uniform/mulMul8conv2d_4/kernel/Initializer/random_uniform/RandomUniform.conv2d_4/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_4/kernel*
T0*&
_output_shapes
: @
ć
*conv2d_4/kernel/Initializer/random_uniformAdd.conv2d_4/kernel/Initializer/random_uniform/mul.conv2d_4/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_4/kernel*
T0*&
_output_shapes
: @
ˇ
conv2d_4/kernel
VariableV2*
	container *&
_output_shapes
: @*
dtype0*
shape: @*"
_class
loc:@conv2d_4/kernel*
shared_name 
Ű
conv2d_4/kernel/AssignAssignconv2d_4/kernel*conv2d_4/kernel/Initializer/random_uniform*
validate_shape(*"
_class
loc:@conv2d_4/kernel*
use_locking(*
T0*&
_output_shapes
: @

conv2d_4/kernel/readIdentityconv2d_4/kernel*"
_class
loc:@conv2d_4/kernel*
T0*&
_output_shapes
: @

conv2d_4/bias/Initializer/zerosConst*
dtype0* 
_class
loc:@conv2d_4/bias*
valueB@*    *
_output_shapes
:@

conv2d_4/bias
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@* 
_class
loc:@conv2d_4/bias*
shared_name 
ž
conv2d_4/bias/AssignAssignconv2d_4/biasconv2d_4/bias/Initializer/zeros*
validate_shape(* 
_class
loc:@conv2d_4/bias*
use_locking(*
T0*
_output_shapes
:@
t
conv2d_4/bias/readIdentityconv2d_4/bias* 
_class
loc:@conv2d_4/bias*
T0*
_output_shapes
:@
g
conv2d_5/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
î
conv2d_5/Conv2DConv2Dmax_pooling2d/MaxPoolconv2d_4/kernel/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0*
	dilations


conv2d_5/BiasAddBiasAddconv2d_5/Conv2Dconv2d_4/bias/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
data_formatNHWC*
T0
a
conv2d_5/ReluReluconv2d_5/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
­
0conv2d_5/kernel/Initializer/random_uniform/shapeConst*
dtype0*"
_class
loc:@conv2d_5/kernel*%
valueB"          @   *
_output_shapes
:

.conv2d_5/kernel/Initializer/random_uniform/minConst*
dtype0*"
_class
loc:@conv2d_5/kernel*
valueB
 *ÍĚL˝*
_output_shapes
: 

.conv2d_5/kernel/Initializer/random_uniform/maxConst*
dtype0*"
_class
loc:@conv2d_5/kernel*
valueB
 *ÍĚL=*
_output_shapes
: 
ö
8conv2d_5/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_5/kernel/Initializer/random_uniform/shape*&
_output_shapes
: @*
dtype0*
seed2 *

seed *
T0*"
_class
loc:@conv2d_5/kernel
Ú
.conv2d_5/kernel/Initializer/random_uniform/subSub.conv2d_5/kernel/Initializer/random_uniform/max.conv2d_5/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_5/kernel*
T0*
_output_shapes
: 
ô
.conv2d_5/kernel/Initializer/random_uniform/mulMul8conv2d_5/kernel/Initializer/random_uniform/RandomUniform.conv2d_5/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_5/kernel*
T0*&
_output_shapes
: @
ć
*conv2d_5/kernel/Initializer/random_uniformAdd.conv2d_5/kernel/Initializer/random_uniform/mul.conv2d_5/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_5/kernel*
T0*&
_output_shapes
: @
ˇ
conv2d_5/kernel
VariableV2*
	container *&
_output_shapes
: @*
dtype0*
shape: @*"
_class
loc:@conv2d_5/kernel*
shared_name 
Ű
conv2d_5/kernel/AssignAssignconv2d_5/kernel*conv2d_5/kernel/Initializer/random_uniform*
validate_shape(*"
_class
loc:@conv2d_5/kernel*
use_locking(*
T0*&
_output_shapes
: @

conv2d_5/kernel/readIdentityconv2d_5/kernel*"
_class
loc:@conv2d_5/kernel*
T0*&
_output_shapes
: @

conv2d_5/bias/Initializer/zerosConst*
dtype0* 
_class
loc:@conv2d_5/bias*
valueB@*    *
_output_shapes
:@

conv2d_5/bias
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@* 
_class
loc:@conv2d_5/bias*
shared_name 
ž
conv2d_5/bias/AssignAssignconv2d_5/biasconv2d_5/bias/Initializer/zeros*
validate_shape(* 
_class
loc:@conv2d_5/bias*
use_locking(*
T0*
_output_shapes
:@
t
conv2d_5/bias/readIdentityconv2d_5/bias* 
_class
loc:@conv2d_5/bias*
T0*
_output_shapes
:@
g
conv2d_6/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
đ
conv2d_6/Conv2DConv2Dmax_pooling2d_2/MaxPoolconv2d_5/kernel/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0*
	dilations


conv2d_6/BiasAddBiasAddconv2d_6/Conv2Dconv2d_5/bias/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
data_formatNHWC*
T0
a
conv2d_6/ReluReluconv2d_6/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
­
0conv2d_6/kernel/Initializer/random_uniform/shapeConst*
dtype0*"
_class
loc:@conv2d_6/kernel*%
valueB"          @   *
_output_shapes
:

.conv2d_6/kernel/Initializer/random_uniform/minConst*
dtype0*"
_class
loc:@conv2d_6/kernel*
valueB
 *ÍĚL˝*
_output_shapes
: 

.conv2d_6/kernel/Initializer/random_uniform/maxConst*
dtype0*"
_class
loc:@conv2d_6/kernel*
valueB
 *ÍĚL=*
_output_shapes
: 
ö
8conv2d_6/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_6/kernel/Initializer/random_uniform/shape*&
_output_shapes
: @*
dtype0*
seed2 *

seed *
T0*"
_class
loc:@conv2d_6/kernel
Ú
.conv2d_6/kernel/Initializer/random_uniform/subSub.conv2d_6/kernel/Initializer/random_uniform/max.conv2d_6/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_6/kernel*
T0*
_output_shapes
: 
ô
.conv2d_6/kernel/Initializer/random_uniform/mulMul8conv2d_6/kernel/Initializer/random_uniform/RandomUniform.conv2d_6/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_6/kernel*
T0*&
_output_shapes
: @
ć
*conv2d_6/kernel/Initializer/random_uniformAdd.conv2d_6/kernel/Initializer/random_uniform/mul.conv2d_6/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_6/kernel*
T0*&
_output_shapes
: @
ˇ
conv2d_6/kernel
VariableV2*
	container *&
_output_shapes
: @*
dtype0*
shape: @*"
_class
loc:@conv2d_6/kernel*
shared_name 
Ű
conv2d_6/kernel/AssignAssignconv2d_6/kernel*conv2d_6/kernel/Initializer/random_uniform*
validate_shape(*"
_class
loc:@conv2d_6/kernel*
use_locking(*
T0*&
_output_shapes
: @

conv2d_6/kernel/readIdentityconv2d_6/kernel*"
_class
loc:@conv2d_6/kernel*
T0*&
_output_shapes
: @

conv2d_6/bias/Initializer/zerosConst*
dtype0* 
_class
loc:@conv2d_6/bias*
valueB@*    *
_output_shapes
:@

conv2d_6/bias
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@* 
_class
loc:@conv2d_6/bias*
shared_name 
ž
conv2d_6/bias/AssignAssignconv2d_6/biasconv2d_6/bias/Initializer/zeros*
validate_shape(* 
_class
loc:@conv2d_6/bias*
use_locking(*
T0*
_output_shapes
:@
t
conv2d_6/bias/readIdentityconv2d_6/bias* 
_class
loc:@conv2d_6/bias*
T0*
_output_shapes
:@
g
conv2d_7/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
đ
conv2d_7/Conv2DConv2Dmax_pooling2d_3/MaxPoolconv2d_6/kernel/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0*
	dilations


conv2d_7/BiasAddBiasAddconv2d_7/Conv2Dconv2d_6/bias/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
data_formatNHWC*
T0
a
conv2d_7/ReluReluconv2d_7/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
­
0conv2d_7/kernel/Initializer/random_uniform/shapeConst*
dtype0*"
_class
loc:@conv2d_7/kernel*%
valueB"          @   *
_output_shapes
:

.conv2d_7/kernel/Initializer/random_uniform/minConst*
dtype0*"
_class
loc:@conv2d_7/kernel*
valueB
 *ÍĚL˝*
_output_shapes
: 

.conv2d_7/kernel/Initializer/random_uniform/maxConst*
dtype0*"
_class
loc:@conv2d_7/kernel*
valueB
 *ÍĚL=*
_output_shapes
: 
ö
8conv2d_7/kernel/Initializer/random_uniform/RandomUniformRandomUniform0conv2d_7/kernel/Initializer/random_uniform/shape*&
_output_shapes
: @*
dtype0*
seed2 *

seed *
T0*"
_class
loc:@conv2d_7/kernel
Ú
.conv2d_7/kernel/Initializer/random_uniform/subSub.conv2d_7/kernel/Initializer/random_uniform/max.conv2d_7/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_7/kernel*
T0*
_output_shapes
: 
ô
.conv2d_7/kernel/Initializer/random_uniform/mulMul8conv2d_7/kernel/Initializer/random_uniform/RandomUniform.conv2d_7/kernel/Initializer/random_uniform/sub*"
_class
loc:@conv2d_7/kernel*
T0*&
_output_shapes
: @
ć
*conv2d_7/kernel/Initializer/random_uniformAdd.conv2d_7/kernel/Initializer/random_uniform/mul.conv2d_7/kernel/Initializer/random_uniform/min*"
_class
loc:@conv2d_7/kernel*
T0*&
_output_shapes
: @
ˇ
conv2d_7/kernel
VariableV2*
	container *&
_output_shapes
: @*
dtype0*
shape: @*"
_class
loc:@conv2d_7/kernel*
shared_name 
Ű
conv2d_7/kernel/AssignAssignconv2d_7/kernel*conv2d_7/kernel/Initializer/random_uniform*
validate_shape(*"
_class
loc:@conv2d_7/kernel*
use_locking(*
T0*&
_output_shapes
: @

conv2d_7/kernel/readIdentityconv2d_7/kernel*"
_class
loc:@conv2d_7/kernel*
T0*&
_output_shapes
: @

conv2d_7/bias/Initializer/zerosConst*
dtype0* 
_class
loc:@conv2d_7/bias*
valueB@*    *
_output_shapes
:@

conv2d_7/bias
VariableV2*
	container *
_output_shapes
:@*
dtype0*
shape:@* 
_class
loc:@conv2d_7/bias*
shared_name 
ž
conv2d_7/bias/AssignAssignconv2d_7/biasconv2d_7/bias/Initializer/zeros*
validate_shape(* 
_class
loc:@conv2d_7/bias*
use_locking(*
T0*
_output_shapes
:@
t
conv2d_7/bias/readIdentityconv2d_7/bias* 
_class
loc:@conv2d_7/bias*
T0*
_output_shapes
:@
g
conv2d_8/dilation_rateConst*
dtype0*
valueB"      *
_output_shapes
:
đ
conv2d_8/Conv2DConv2Dmax_pooling2d_4/MaxPoolconv2d_7/kernel/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
data_formatNHWC*
use_cudnn_on_gpu(*
paddingSAME*
strides
*
T0*
	dilations


conv2d_8/BiasAddBiasAddconv2d_8/Conv2Dconv2d_7/bias/read*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
data_formatNHWC*
T0
a
conv2d_8/ReluReluconv2d_8/BiasAdd*
T0*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@
ž
max_pooling2d_5/MaxPoolMaxPoolconv2d_5/Relu*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
data_formatNHWC*
paddingVALID*
strides
*
ksize
*
T0
ž
max_pooling2d_6/MaxPoolMaxPoolconv2d_6/Relu*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
data_formatNHWC*
paddingVALID*
strides
*
ksize
*
T0
ž
max_pooling2d_7/MaxPoolMaxPoolconv2d_7/Relu*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
data_formatNHWC*
paddingVALID*
strides
*
ksize
*
T0
ž
max_pooling2d_8/MaxPoolMaxPoolconv2d_8/Relu*/
_output_shapes
:˙˙˙˙˙˙˙˙˙@*
data_formatNHWC*
paddingVALID*
strides
*
ksize
*
T0
`
Reshape_1/shapeConst*
dtype0*
valueB"˙˙˙˙@  *
_output_shapes
:

	Reshape_1Reshapemax_pooling2d_5/MaxPoolReshape_1/shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ*
Tshape0*
T0
`
Reshape_2/shapeConst*
dtype0*
valueB"˙˙˙˙@  *
_output_shapes
:

	Reshape_2Reshapemax_pooling2d_6/MaxPoolReshape_2/shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ*
Tshape0*
T0
`
Reshape_3/shapeConst*
dtype0*
valueB"˙˙˙˙@  *
_output_shapes
:

	Reshape_3Reshapemax_pooling2d_7/MaxPoolReshape_3/shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ*
Tshape0*
T0
`
Reshape_4/shapeConst*
dtype0*
valueB"˙˙˙˙@  *
_output_shapes
:

	Reshape_4Reshapemax_pooling2d_8/MaxPoolReshape_4/shape*(
_output_shapes
:˙˙˙˙˙˙˙˙˙Ŕ*
Tshape0*
T0

-dense/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_class
loc:@dense/kernel*
valueB"@     *
_output_shapes
:

+dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_class
loc:@dense/kernel*
valueB
 *˝*
_output_shapes
: 

+dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
_class
loc:@dense/kernel*
valueB
 *=*
_output_shapes
: 
ç
5dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform-dense/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
Ŕ*
dtype0*
seed2 *

seed *
T0*
_class
loc:@dense/kernel
Î
+dense/kernel/Initializer/random_uniform/subSub+dense/kernel/Initializer/random_uniform/max+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
T0*
_output_shapes
: 
â
+dense/kernel/Initializer/random_uniform/mulMul5dense/kernel/Initializer/random_uniform/RandomUniform+dense/kernel/Initializer/random_uniform/sub*
_class
loc:@dense/kernel*
T0* 
_output_shapes
:
Ŕ
Ô
'dense/kernel/Initializer/random_uniformAdd+dense/kernel/Initializer/random_uniform/mul+dense/kernel/Initializer/random_uniform/min*
_class
loc:@dense/kernel*
T0* 
_output_shapes
:
Ŕ
Ľ
dense/kernel
VariableV2*
	container * 
_output_shapes
:
Ŕ*
dtype0*
shape:
Ŕ*
_class
loc:@dense/kernel*
shared_name 
É
dense/kernel/AssignAssigndense/kernel'dense/kernel/Initializer/random_uniform*
validate_shape(*
_class
loc:@dense/kernel*
use_locking(*
T0* 
_output_shapes
:
Ŕ
w
dense/kernel/readIdentitydense/kernel*
_class
loc:@dense/kernel*
T0* 
_output_shapes
:
Ŕ

dense/bias/Initializer/zerosConst*
dtype0*
_class
loc:@dense/bias*
valueB*    *
_output_shapes	
:


dense/bias
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*
_class
loc:@dense/bias*
shared_name 
ł
dense/bias/AssignAssign
dense/biasdense/bias/Initializer/zeros*
validate_shape(*
_class
loc:@dense/bias*
use_locking(*
T0*
_output_shapes	
:
l
dense/bias/readIdentity
dense/bias*
_class
loc:@dense/bias*
T0*
_output_shapes	
:

dense/MatMulMatMul	Reshape_1dense/kernel/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dense/BiasAddBiasAdddense/MatMuldense/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
T

dense/ReluReludense/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
/dense_1/kernel/Initializer/random_uniform/shapeConst*
dtype0*!
_class
loc:@dense_1/kernel*
valueB"@     *
_output_shapes
:

-dense_1/kernel/Initializer/random_uniform/minConst*
dtype0*!
_class
loc:@dense_1/kernel*
valueB
 *˝*
_output_shapes
: 

-dense_1/kernel/Initializer/random_uniform/maxConst*
dtype0*!
_class
loc:@dense_1/kernel*
valueB
 *=*
_output_shapes
: 
í
7dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_1/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
Ŕ*
dtype0*
seed2 *

seed *
T0*!
_class
loc:@dense_1/kernel
Ö
-dense_1/kernel/Initializer/random_uniform/subSub-dense_1/kernel/Initializer/random_uniform/max-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
T0*
_output_shapes
: 
ę
-dense_1/kernel/Initializer/random_uniform/mulMul7dense_1/kernel/Initializer/random_uniform/RandomUniform-dense_1/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_1/kernel*
T0* 
_output_shapes
:
Ŕ
Ü
)dense_1/kernel/Initializer/random_uniformAdd-dense_1/kernel/Initializer/random_uniform/mul-dense_1/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_1/kernel*
T0* 
_output_shapes
:
Ŕ
Š
dense_1/kernel
VariableV2*
	container * 
_output_shapes
:
Ŕ*
dtype0*
shape:
Ŕ*!
_class
loc:@dense_1/kernel*
shared_name 
Ń
dense_1/kernel/AssignAssigndense_1/kernel)dense_1/kernel/Initializer/random_uniform*
validate_shape(*!
_class
loc:@dense_1/kernel*
use_locking(*
T0* 
_output_shapes
:
Ŕ
}
dense_1/kernel/readIdentitydense_1/kernel*!
_class
loc:@dense_1/kernel*
T0* 
_output_shapes
:
Ŕ

dense_1/bias/Initializer/zerosConst*
dtype0*
_class
loc:@dense_1/bias*
valueB*    *
_output_shapes	
:

dense_1/bias
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*
_class
loc:@dense_1/bias*
shared_name 
ť
dense_1/bias/AssignAssigndense_1/biasdense_1/bias/Initializer/zeros*
validate_shape(*
_class
loc:@dense_1/bias*
use_locking(*
T0*
_output_shapes	
:
r
dense_1/bias/readIdentitydense_1/bias*
_class
loc:@dense_1/bias*
T0*
_output_shapes	
:

dense_2/MatMulMatMul	Reshape_2dense_1/kernel/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dense_2/BiasAddBiasAdddense_2/MatMuldense_1/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
X
dense_2/ReluReludense_2/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
/dense_2/kernel/Initializer/random_uniform/shapeConst*
dtype0*!
_class
loc:@dense_2/kernel*
valueB"@     *
_output_shapes
:

-dense_2/kernel/Initializer/random_uniform/minConst*
dtype0*!
_class
loc:@dense_2/kernel*
valueB
 *˝*
_output_shapes
: 

-dense_2/kernel/Initializer/random_uniform/maxConst*
dtype0*!
_class
loc:@dense_2/kernel*
valueB
 *=*
_output_shapes
: 
í
7dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_2/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
Ŕ*
dtype0*
seed2 *

seed *
T0*!
_class
loc:@dense_2/kernel
Ö
-dense_2/kernel/Initializer/random_uniform/subSub-dense_2/kernel/Initializer/random_uniform/max-dense_2/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_2/kernel*
T0*
_output_shapes
: 
ę
-dense_2/kernel/Initializer/random_uniform/mulMul7dense_2/kernel/Initializer/random_uniform/RandomUniform-dense_2/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_2/kernel*
T0* 
_output_shapes
:
Ŕ
Ü
)dense_2/kernel/Initializer/random_uniformAdd-dense_2/kernel/Initializer/random_uniform/mul-dense_2/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_2/kernel*
T0* 
_output_shapes
:
Ŕ
Š
dense_2/kernel
VariableV2*
	container * 
_output_shapes
:
Ŕ*
dtype0*
shape:
Ŕ*!
_class
loc:@dense_2/kernel*
shared_name 
Ń
dense_2/kernel/AssignAssigndense_2/kernel)dense_2/kernel/Initializer/random_uniform*
validate_shape(*!
_class
loc:@dense_2/kernel*
use_locking(*
T0* 
_output_shapes
:
Ŕ
}
dense_2/kernel/readIdentitydense_2/kernel*!
_class
loc:@dense_2/kernel*
T0* 
_output_shapes
:
Ŕ

dense_2/bias/Initializer/zerosConst*
dtype0*
_class
loc:@dense_2/bias*
valueB*    *
_output_shapes	
:

dense_2/bias
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*
_class
loc:@dense_2/bias*
shared_name 
ť
dense_2/bias/AssignAssigndense_2/biasdense_2/bias/Initializer/zeros*
validate_shape(*
_class
loc:@dense_2/bias*
use_locking(*
T0*
_output_shapes	
:
r
dense_2/bias/readIdentitydense_2/bias*
_class
loc:@dense_2/bias*
T0*
_output_shapes	
:

dense_3/MatMulMatMul	Reshape_3dense_2/kernel/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dense_3/BiasAddBiasAdddense_3/MatMuldense_2/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
X
dense_3/ReluReludense_3/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
/dense_3/kernel/Initializer/random_uniform/shapeConst*
dtype0*!
_class
loc:@dense_3/kernel*
valueB"@     *
_output_shapes
:

-dense_3/kernel/Initializer/random_uniform/minConst*
dtype0*!
_class
loc:@dense_3/kernel*
valueB
 *˝*
_output_shapes
: 

-dense_3/kernel/Initializer/random_uniform/maxConst*
dtype0*!
_class
loc:@dense_3/kernel*
valueB
 *=*
_output_shapes
: 
í
7dense_3/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_3/kernel/Initializer/random_uniform/shape* 
_output_shapes
:
Ŕ*
dtype0*
seed2 *

seed *
T0*!
_class
loc:@dense_3/kernel
Ö
-dense_3/kernel/Initializer/random_uniform/subSub-dense_3/kernel/Initializer/random_uniform/max-dense_3/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_3/kernel*
T0*
_output_shapes
: 
ę
-dense_3/kernel/Initializer/random_uniform/mulMul7dense_3/kernel/Initializer/random_uniform/RandomUniform-dense_3/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_3/kernel*
T0* 
_output_shapes
:
Ŕ
Ü
)dense_3/kernel/Initializer/random_uniformAdd-dense_3/kernel/Initializer/random_uniform/mul-dense_3/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_3/kernel*
T0* 
_output_shapes
:
Ŕ
Š
dense_3/kernel
VariableV2*
	container * 
_output_shapes
:
Ŕ*
dtype0*
shape:
Ŕ*!
_class
loc:@dense_3/kernel*
shared_name 
Ń
dense_3/kernel/AssignAssigndense_3/kernel)dense_3/kernel/Initializer/random_uniform*
validate_shape(*!
_class
loc:@dense_3/kernel*
use_locking(*
T0* 
_output_shapes
:
Ŕ
}
dense_3/kernel/readIdentitydense_3/kernel*!
_class
loc:@dense_3/kernel*
T0* 
_output_shapes
:
Ŕ

dense_3/bias/Initializer/zerosConst*
dtype0*
_class
loc:@dense_3/bias*
valueB*    *
_output_shapes	
:

dense_3/bias
VariableV2*
	container *
_output_shapes	
:*
dtype0*
shape:*
_class
loc:@dense_3/bias*
shared_name 
ť
dense_3/bias/AssignAssigndense_3/biasdense_3/bias/Initializer/zeros*
validate_shape(*
_class
loc:@dense_3/bias*
use_locking(*
T0*
_output_shapes	
:
r
dense_3/bias/readIdentitydense_3/bias*
_class
loc:@dense_3/bias*
T0*
_output_shapes	
:

dense_4/MatMulMatMul	Reshape_4dense_3/kernel/read*
transpose_b( *
transpose_a( *
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙

dense_4/BiasAddBiasAdddense_4/MatMuldense_3/bias/read*(
_output_shapes
:˙˙˙˙˙˙˙˙˙*
data_formatNHWC*
T0
X
dense_4/ReluReludense_4/BiasAdd*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
[
dropout/IdentityIdentity
dense/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
dropout_2/IdentityIdentitydense_2/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
dropout_3/IdentityIdentitydense_3/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
_
dropout_4/IdentityIdentitydense_4/Relu*
T0*(
_output_shapes
:˙˙˙˙˙˙˙˙˙
Ł
/dense_4/kernel/Initializer/random_uniform/shapeConst*
dtype0*!
_class
loc:@dense_4/kernel*
valueB"   
   *
_output_shapes
:

-dense_4/kernel/Initializer/random_uniform/minConst*
dtype0*!
_class
loc:@dense_4/kernel*
valueB
 *č˝*
_output_shapes
: 

-dense_4/kernel/Initializer/random_uniform/maxConst*
dtype0*!
_class
loc:@dense_4/kernel*
valueB
 *č=*
_output_shapes
: 
ě
7dense_4/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_4/kernel/Initializer/random_uniform/shape*
_output_shapes
:	
*
dtype0*
seed2 *

seed *
T0*!
_class
loc:@dense_4/kernel
Ö
-dense_4/kernel/Initializer/random_uniform/subSub-dense_4/kernel/Initializer/random_uniform/max-dense_4/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_4/kernel*
T0*
_output_shapes
: 
é
-dense_4/kernel/Initializer/random_uniform/mulMul7dense_4/kernel/Initializer/random_uniform/RandomUniform-dense_4/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_4/kernel*
T0*
_output_shapes
:	

Ű
)dense_4/kernel/Initializer/random_uniformAdd-dense_4/kernel/Initializer/random_uniform/mul-dense_4/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_4/kernel*
T0*
_output_shapes
:	

§
dense_4/kernel
VariableV2*
	container *
_output_shapes
:	
*
dtype0*
shape:	
*!
_class
loc:@dense_4/kernel*
shared_name 
Đ
dense_4/kernel/AssignAssigndense_4/kernel)dense_4/kernel/Initializer/random_uniform*
validate_shape(*!
_class
loc:@dense_4/kernel*
use_locking(*
T0*
_output_shapes
:	

|
dense_4/kernel/readIdentitydense_4/kernel*!
_class
loc:@dense_4/kernel*
T0*
_output_shapes
:	


dense_4/bias/Initializer/zerosConst*
dtype0*
_class
loc:@dense_4/bias*
valueB
*    *
_output_shapes
:


dense_4/bias
VariableV2*
	container *
_output_shapes
:
*
dtype0*
shape:
*
_class
loc:@dense_4/bias*
shared_name 
ş
dense_4/bias/AssignAssigndense_4/biasdense_4/bias/Initializer/zeros*
validate_shape(*
_class
loc:@dense_4/bias*
use_locking(*
T0*
_output_shapes
:

q
dense_4/bias/readIdentitydense_4/bias*
_class
loc:@dense_4/bias*
T0*
_output_shapes
:


dense_5/MatMulMatMuldropout/Identitydense_4/kernel/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙


dense_5/BiasAddBiasAdddense_5/MatMuldense_4/bias/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
data_formatNHWC*
T0
Ł
/dense_5/kernel/Initializer/random_uniform/shapeConst*
dtype0*!
_class
loc:@dense_5/kernel*
valueB"   
   *
_output_shapes
:

-dense_5/kernel/Initializer/random_uniform/minConst*
dtype0*!
_class
loc:@dense_5/kernel*
valueB
 *č˝*
_output_shapes
: 

-dense_5/kernel/Initializer/random_uniform/maxConst*
dtype0*!
_class
loc:@dense_5/kernel*
valueB
 *č=*
_output_shapes
: 
ě
7dense_5/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_5/kernel/Initializer/random_uniform/shape*
_output_shapes
:	
*
dtype0*
seed2 *

seed *
T0*!
_class
loc:@dense_5/kernel
Ö
-dense_5/kernel/Initializer/random_uniform/subSub-dense_5/kernel/Initializer/random_uniform/max-dense_5/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_5/kernel*
T0*
_output_shapes
: 
é
-dense_5/kernel/Initializer/random_uniform/mulMul7dense_5/kernel/Initializer/random_uniform/RandomUniform-dense_5/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_5/kernel*
T0*
_output_shapes
:	

Ű
)dense_5/kernel/Initializer/random_uniformAdd-dense_5/kernel/Initializer/random_uniform/mul-dense_5/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_5/kernel*
T0*
_output_shapes
:	

§
dense_5/kernel
VariableV2*
	container *
_output_shapes
:	
*
dtype0*
shape:	
*!
_class
loc:@dense_5/kernel*
shared_name 
Đ
dense_5/kernel/AssignAssigndense_5/kernel)dense_5/kernel/Initializer/random_uniform*
validate_shape(*!
_class
loc:@dense_5/kernel*
use_locking(*
T0*
_output_shapes
:	

|
dense_5/kernel/readIdentitydense_5/kernel*!
_class
loc:@dense_5/kernel*
T0*
_output_shapes
:	


dense_5/bias/Initializer/zerosConst*
dtype0*
_class
loc:@dense_5/bias*
valueB
*    *
_output_shapes
:


dense_5/bias
VariableV2*
	container *
_output_shapes
:
*
dtype0*
shape:
*
_class
loc:@dense_5/bias*
shared_name 
ş
dense_5/bias/AssignAssigndense_5/biasdense_5/bias/Initializer/zeros*
validate_shape(*
_class
loc:@dense_5/bias*
use_locking(*
T0*
_output_shapes
:

q
dense_5/bias/readIdentitydense_5/bias*
_class
loc:@dense_5/bias*
T0*
_output_shapes
:


dense_6/MatMulMatMuldropout_2/Identitydense_5/kernel/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙


dense_6/BiasAddBiasAdddense_6/MatMuldense_5/bias/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
data_formatNHWC*
T0
Ł
/dense_6/kernel/Initializer/random_uniform/shapeConst*
dtype0*!
_class
loc:@dense_6/kernel*
valueB"   
   *
_output_shapes
:

-dense_6/kernel/Initializer/random_uniform/minConst*
dtype0*!
_class
loc:@dense_6/kernel*
valueB
 *č˝*
_output_shapes
: 

-dense_6/kernel/Initializer/random_uniform/maxConst*
dtype0*!
_class
loc:@dense_6/kernel*
valueB
 *č=*
_output_shapes
: 
ě
7dense_6/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_6/kernel/Initializer/random_uniform/shape*
_output_shapes
:	
*
dtype0*
seed2 *

seed *
T0*!
_class
loc:@dense_6/kernel
Ö
-dense_6/kernel/Initializer/random_uniform/subSub-dense_6/kernel/Initializer/random_uniform/max-dense_6/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_6/kernel*
T0*
_output_shapes
: 
é
-dense_6/kernel/Initializer/random_uniform/mulMul7dense_6/kernel/Initializer/random_uniform/RandomUniform-dense_6/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_6/kernel*
T0*
_output_shapes
:	

Ű
)dense_6/kernel/Initializer/random_uniformAdd-dense_6/kernel/Initializer/random_uniform/mul-dense_6/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_6/kernel*
T0*
_output_shapes
:	

§
dense_6/kernel
VariableV2*
	container *
_output_shapes
:	
*
dtype0*
shape:	
*!
_class
loc:@dense_6/kernel*
shared_name 
Đ
dense_6/kernel/AssignAssigndense_6/kernel)dense_6/kernel/Initializer/random_uniform*
validate_shape(*!
_class
loc:@dense_6/kernel*
use_locking(*
T0*
_output_shapes
:	

|
dense_6/kernel/readIdentitydense_6/kernel*!
_class
loc:@dense_6/kernel*
T0*
_output_shapes
:	


dense_6/bias/Initializer/zerosConst*
dtype0*
_class
loc:@dense_6/bias*
valueB
*    *
_output_shapes
:


dense_6/bias
VariableV2*
	container *
_output_shapes
:
*
dtype0*
shape:
*
_class
loc:@dense_6/bias*
shared_name 
ş
dense_6/bias/AssignAssigndense_6/biasdense_6/bias/Initializer/zeros*
validate_shape(*
_class
loc:@dense_6/bias*
use_locking(*
T0*
_output_shapes
:

q
dense_6/bias/readIdentitydense_6/bias*
_class
loc:@dense_6/bias*
T0*
_output_shapes
:


dense_7/MatMulMatMuldropout_3/Identitydense_6/kernel/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙


dense_7/BiasAddBiasAdddense_7/MatMuldense_6/bias/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
data_formatNHWC*
T0
Ł
/dense_7/kernel/Initializer/random_uniform/shapeConst*
dtype0*!
_class
loc:@dense_7/kernel*
valueB"   
   *
_output_shapes
:

-dense_7/kernel/Initializer/random_uniform/minConst*
dtype0*!
_class
loc:@dense_7/kernel*
valueB
 *č˝*
_output_shapes
: 

-dense_7/kernel/Initializer/random_uniform/maxConst*
dtype0*!
_class
loc:@dense_7/kernel*
valueB
 *č=*
_output_shapes
: 
ě
7dense_7/kernel/Initializer/random_uniform/RandomUniformRandomUniform/dense_7/kernel/Initializer/random_uniform/shape*
_output_shapes
:	
*
dtype0*
seed2 *

seed *
T0*!
_class
loc:@dense_7/kernel
Ö
-dense_7/kernel/Initializer/random_uniform/subSub-dense_7/kernel/Initializer/random_uniform/max-dense_7/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_7/kernel*
T0*
_output_shapes
: 
é
-dense_7/kernel/Initializer/random_uniform/mulMul7dense_7/kernel/Initializer/random_uniform/RandomUniform-dense_7/kernel/Initializer/random_uniform/sub*!
_class
loc:@dense_7/kernel*
T0*
_output_shapes
:	

Ű
)dense_7/kernel/Initializer/random_uniformAdd-dense_7/kernel/Initializer/random_uniform/mul-dense_7/kernel/Initializer/random_uniform/min*!
_class
loc:@dense_7/kernel*
T0*
_output_shapes
:	

§
dense_7/kernel
VariableV2*
	container *
_output_shapes
:	
*
dtype0*
shape:	
*!
_class
loc:@dense_7/kernel*
shared_name 
Đ
dense_7/kernel/AssignAssigndense_7/kernel)dense_7/kernel/Initializer/random_uniform*
validate_shape(*!
_class
loc:@dense_7/kernel*
use_locking(*
T0*
_output_shapes
:	

|
dense_7/kernel/readIdentitydense_7/kernel*!
_class
loc:@dense_7/kernel*
T0*
_output_shapes
:	


dense_7/bias/Initializer/zerosConst*
dtype0*
_class
loc:@dense_7/bias*
valueB
*    *
_output_shapes
:


dense_7/bias
VariableV2*
	container *
_output_shapes
:
*
dtype0*
shape:
*
_class
loc:@dense_7/bias*
shared_name 
ş
dense_7/bias/AssignAssigndense_7/biasdense_7/bias/Initializer/zeros*
validate_shape(*
_class
loc:@dense_7/bias*
use_locking(*
T0*
_output_shapes
:

q
dense_7/bias/readIdentitydense_7/bias*
_class
loc:@dense_7/bias*
T0*
_output_shapes
:


dense_8/MatMulMatMuldropout_4/Identitydense_7/kernel/read*
transpose_b( *
transpose_a( *
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙


dense_8/BiasAddBiasAdddense_8/MatMuldense_7/bias/read*'
_output_shapes
:˙˙˙˙˙˙˙˙˙
*
data_formatNHWC*
T0
^
AddAdddense_5/BiasAdddense_6/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

`
Add_1Adddense_7/BiasAdddense_8/BiasAdd*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

J
Add_2AddAddAdd_1*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

N
	truediv/yConst*
dtype0*
valueB
 *  @*
_output_shapes
: 
V
truedivRealDivAdd_2	truediv/y*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

R
ArgMax/dimensionConst*
dtype0*
value	B :*
_output_shapes
: 
x
ArgMaxArgMaxtruedivArgMax/dimension*
output_type0	*#
_output_shapes
:˙˙˙˙˙˙˙˙˙*

Tidx0*
T0
¨
AsStringAsStringArgMax*

scientific( *#
_output_shapes
:˙˙˙˙˙˙˙˙˙*
	precision˙˙˙˙˙˙˙˙˙*
width˙˙˙˙˙˙˙˙˙*
T0	*
shortest( *

fill 
T
softmax_tensorSoftmaxtruediv*
T0*'
_output_shapes
:˙˙˙˙˙˙˙˙˙

P

save/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 

save/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_1218acb267214af88d5e247abeb4981a/part*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
Q
save/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
k
save/ShardedFilename/shardConst"/device:CPU:0*
dtype0*
value	B : *
_output_shapes
: 

save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards"/device:CPU:0*
_output_shapes
: 
ć
save/SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
valueBý!Bconv2d/biasBconv2d/kernelBconv2d_1/biasBconv2d_1/kernelBconv2d_2/biasBconv2d_2/kernelBconv2d_3/biasBconv2d_3/kernelBconv2d_4/biasBconv2d_4/kernelBconv2d_5/biasBconv2d_5/kernelBconv2d_6/biasBconv2d_6/kernelBconv2d_7/biasBconv2d_7/kernelB
dense/biasBdense/kernelBdense_1/biasBdense_1/kernelBdense_2/biasBdense_2/kernelBdense_3/biasBdense_3/kernelBdense_4/biasBdense_4/kernelBdense_5/biasBdense_5/kernelBdense_6/biasBdense_6/kernelBdense_7/biasBdense_7/kernelBglobal_step*
_output_shapes
:!
´
save/SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:!

save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesconv2d/biasconv2d/kernelconv2d_1/biasconv2d_1/kernelconv2d_2/biasconv2d_2/kernelconv2d_3/biasconv2d_3/kernelconv2d_4/biasconv2d_4/kernelconv2d_5/biasconv2d_5/kernelconv2d_6/biasconv2d_6/kernelconv2d_7/biasconv2d_7/kernel
dense/biasdense/kerneldense_1/biasdense_1/kerneldense_2/biasdense_2/kerneldense_3/biasdense_3/kerneldense_4/biasdense_4/kerneldense_5/biasdense_5/kerneldense_6/biasdense_6/kerneldense_7/biasdense_7/kernelglobal_step"/device:CPU:0*/
dtypes%
#2!	
 
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2"/device:CPU:0*'
_class
loc:@save/ShardedFilename*
T0*
_output_shapes
: 
Ź
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency"/device:CPU:0*
N*

axis *
T0*
_output_shapes
:

save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const"/device:CPU:0*
delete_old_dirs(

save/IdentityIdentity
save/Const^save/control_dependency^save/MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
o
save/RestoreV2/tensor_namesConst*
dtype0* 
valueBBconv2d/bias*
_output_shapes
:
h
save/RestoreV2/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
 
save/AssignAssignconv2d/biassave/RestoreV2*
validate_shape(*
_class
loc:@conv2d/bias*
use_locking(*
T0*
_output_shapes
: 
s
save/RestoreV2_1/tensor_namesConst*
dtype0*"
valueBBconv2d/kernel*
_output_shapes
:
j
!save/RestoreV2_1/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_1	RestoreV2
save/Constsave/RestoreV2_1/tensor_names!save/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
´
save/Assign_1Assignconv2d/kernelsave/RestoreV2_1*
validate_shape(* 
_class
loc:@conv2d/kernel*
use_locking(*
T0*&
_output_shapes
: 
s
save/RestoreV2_2/tensor_namesConst*
dtype0*"
valueBBconv2d_1/bias*
_output_shapes
:
j
!save/RestoreV2_2/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_2	RestoreV2
save/Constsave/RestoreV2_2/tensor_names!save/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
¨
save/Assign_2Assignconv2d_1/biassave/RestoreV2_2*
validate_shape(* 
_class
loc:@conv2d_1/bias*
use_locking(*
T0*
_output_shapes
: 
u
save/RestoreV2_3/tensor_namesConst*
dtype0*$
valueBBconv2d_1/kernel*
_output_shapes
:
j
!save/RestoreV2_3/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_3	RestoreV2
save/Constsave/RestoreV2_3/tensor_names!save/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
¸
save/Assign_3Assignconv2d_1/kernelsave/RestoreV2_3*
validate_shape(*"
_class
loc:@conv2d_1/kernel*
use_locking(*
T0*&
_output_shapes
: 
s
save/RestoreV2_4/tensor_namesConst*
dtype0*"
valueBBconv2d_2/bias*
_output_shapes
:
j
!save/RestoreV2_4/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_4	RestoreV2
save/Constsave/RestoreV2_4/tensor_names!save/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
¨
save/Assign_4Assignconv2d_2/biassave/RestoreV2_4*
validate_shape(* 
_class
loc:@conv2d_2/bias*
use_locking(*
T0*
_output_shapes
: 
u
save/RestoreV2_5/tensor_namesConst*
dtype0*$
valueBBconv2d_2/kernel*
_output_shapes
:
j
!save/RestoreV2_5/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_5	RestoreV2
save/Constsave/RestoreV2_5/tensor_names!save/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
¸
save/Assign_5Assignconv2d_2/kernelsave/RestoreV2_5*
validate_shape(*"
_class
loc:@conv2d_2/kernel*
use_locking(*
T0*&
_output_shapes
: 
s
save/RestoreV2_6/tensor_namesConst*
dtype0*"
valueBBconv2d_3/bias*
_output_shapes
:
j
!save/RestoreV2_6/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_6	RestoreV2
save/Constsave/RestoreV2_6/tensor_names!save/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
¨
save/Assign_6Assignconv2d_3/biassave/RestoreV2_6*
validate_shape(* 
_class
loc:@conv2d_3/bias*
use_locking(*
T0*
_output_shapes
: 
u
save/RestoreV2_7/tensor_namesConst*
dtype0*$
valueBBconv2d_3/kernel*
_output_shapes
:
j
!save/RestoreV2_7/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_7	RestoreV2
save/Constsave/RestoreV2_7/tensor_names!save/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
¸
save/Assign_7Assignconv2d_3/kernelsave/RestoreV2_7*
validate_shape(*"
_class
loc:@conv2d_3/kernel*
use_locking(*
T0*&
_output_shapes
: 
s
save/RestoreV2_8/tensor_namesConst*
dtype0*"
valueBBconv2d_4/bias*
_output_shapes
:
j
!save/RestoreV2_8/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_8	RestoreV2
save/Constsave/RestoreV2_8/tensor_names!save/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
¨
save/Assign_8Assignconv2d_4/biassave/RestoreV2_8*
validate_shape(* 
_class
loc:@conv2d_4/bias*
use_locking(*
T0*
_output_shapes
:@
u
save/RestoreV2_9/tensor_namesConst*
dtype0*$
valueBBconv2d_4/kernel*
_output_shapes
:
j
!save/RestoreV2_9/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_9	RestoreV2
save/Constsave/RestoreV2_9/tensor_names!save/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
¸
save/Assign_9Assignconv2d_4/kernelsave/RestoreV2_9*
validate_shape(*"
_class
loc:@conv2d_4/kernel*
use_locking(*
T0*&
_output_shapes
: @
t
save/RestoreV2_10/tensor_namesConst*
dtype0*"
valueBBconv2d_5/bias*
_output_shapes
:
k
"save/RestoreV2_10/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_10	RestoreV2
save/Constsave/RestoreV2_10/tensor_names"save/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
Ş
save/Assign_10Assignconv2d_5/biassave/RestoreV2_10*
validate_shape(* 
_class
loc:@conv2d_5/bias*
use_locking(*
T0*
_output_shapes
:@
v
save/RestoreV2_11/tensor_namesConst*
dtype0*$
valueBBconv2d_5/kernel*
_output_shapes
:
k
"save/RestoreV2_11/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_11	RestoreV2
save/Constsave/RestoreV2_11/tensor_names"save/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
ş
save/Assign_11Assignconv2d_5/kernelsave/RestoreV2_11*
validate_shape(*"
_class
loc:@conv2d_5/kernel*
use_locking(*
T0*&
_output_shapes
: @
t
save/RestoreV2_12/tensor_namesConst*
dtype0*"
valueBBconv2d_6/bias*
_output_shapes
:
k
"save/RestoreV2_12/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_12	RestoreV2
save/Constsave/RestoreV2_12/tensor_names"save/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
Ş
save/Assign_12Assignconv2d_6/biassave/RestoreV2_12*
validate_shape(* 
_class
loc:@conv2d_6/bias*
use_locking(*
T0*
_output_shapes
:@
v
save/RestoreV2_13/tensor_namesConst*
dtype0*$
valueBBconv2d_6/kernel*
_output_shapes
:
k
"save/RestoreV2_13/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_13	RestoreV2
save/Constsave/RestoreV2_13/tensor_names"save/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
ş
save/Assign_13Assignconv2d_6/kernelsave/RestoreV2_13*
validate_shape(*"
_class
loc:@conv2d_6/kernel*
use_locking(*
T0*&
_output_shapes
: @
t
save/RestoreV2_14/tensor_namesConst*
dtype0*"
valueBBconv2d_7/bias*
_output_shapes
:
k
"save/RestoreV2_14/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_14	RestoreV2
save/Constsave/RestoreV2_14/tensor_names"save/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
Ş
save/Assign_14Assignconv2d_7/biassave/RestoreV2_14*
validate_shape(* 
_class
loc:@conv2d_7/bias*
use_locking(*
T0*
_output_shapes
:@
v
save/RestoreV2_15/tensor_namesConst*
dtype0*$
valueBBconv2d_7/kernel*
_output_shapes
:
k
"save/RestoreV2_15/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_15	RestoreV2
save/Constsave/RestoreV2_15/tensor_names"save/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
ş
save/Assign_15Assignconv2d_7/kernelsave/RestoreV2_15*
validate_shape(*"
_class
loc:@conv2d_7/kernel*
use_locking(*
T0*&
_output_shapes
: @
q
save/RestoreV2_16/tensor_namesConst*
dtype0*
valueBB
dense/bias*
_output_shapes
:
k
"save/RestoreV2_16/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_16	RestoreV2
save/Constsave/RestoreV2_16/tensor_names"save/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
Ľ
save/Assign_16Assign
dense/biassave/RestoreV2_16*
validate_shape(*
_class
loc:@dense/bias*
use_locking(*
T0*
_output_shapes	
:
s
save/RestoreV2_17/tensor_namesConst*
dtype0*!
valueBBdense/kernel*
_output_shapes
:
k
"save/RestoreV2_17/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_17	RestoreV2
save/Constsave/RestoreV2_17/tensor_names"save/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
Ž
save/Assign_17Assigndense/kernelsave/RestoreV2_17*
validate_shape(*
_class
loc:@dense/kernel*
use_locking(*
T0* 
_output_shapes
:
Ŕ
s
save/RestoreV2_18/tensor_namesConst*
dtype0*!
valueBBdense_1/bias*
_output_shapes
:
k
"save/RestoreV2_18/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_18	RestoreV2
save/Constsave/RestoreV2_18/tensor_names"save/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
Š
save/Assign_18Assigndense_1/biassave/RestoreV2_18*
validate_shape(*
_class
loc:@dense_1/bias*
use_locking(*
T0*
_output_shapes	
:
u
save/RestoreV2_19/tensor_namesConst*
dtype0*#
valueBBdense_1/kernel*
_output_shapes
:
k
"save/RestoreV2_19/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_19	RestoreV2
save/Constsave/RestoreV2_19/tensor_names"save/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
˛
save/Assign_19Assigndense_1/kernelsave/RestoreV2_19*
validate_shape(*!
_class
loc:@dense_1/kernel*
use_locking(*
T0* 
_output_shapes
:
Ŕ
s
save/RestoreV2_20/tensor_namesConst*
dtype0*!
valueBBdense_2/bias*
_output_shapes
:
k
"save/RestoreV2_20/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_20	RestoreV2
save/Constsave/RestoreV2_20/tensor_names"save/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
Š
save/Assign_20Assigndense_2/biassave/RestoreV2_20*
validate_shape(*
_class
loc:@dense_2/bias*
use_locking(*
T0*
_output_shapes	
:
u
save/RestoreV2_21/tensor_namesConst*
dtype0*#
valueBBdense_2/kernel*
_output_shapes
:
k
"save/RestoreV2_21/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_21	RestoreV2
save/Constsave/RestoreV2_21/tensor_names"save/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
˛
save/Assign_21Assigndense_2/kernelsave/RestoreV2_21*
validate_shape(*!
_class
loc:@dense_2/kernel*
use_locking(*
T0* 
_output_shapes
:
Ŕ
s
save/RestoreV2_22/tensor_namesConst*
dtype0*!
valueBBdense_3/bias*
_output_shapes
:
k
"save/RestoreV2_22/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_22	RestoreV2
save/Constsave/RestoreV2_22/tensor_names"save/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
Š
save/Assign_22Assigndense_3/biassave/RestoreV2_22*
validate_shape(*
_class
loc:@dense_3/bias*
use_locking(*
T0*
_output_shapes	
:
u
save/RestoreV2_23/tensor_namesConst*
dtype0*#
valueBBdense_3/kernel*
_output_shapes
:
k
"save/RestoreV2_23/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_23	RestoreV2
save/Constsave/RestoreV2_23/tensor_names"save/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
˛
save/Assign_23Assigndense_3/kernelsave/RestoreV2_23*
validate_shape(*!
_class
loc:@dense_3/kernel*
use_locking(*
T0* 
_output_shapes
:
Ŕ
s
save/RestoreV2_24/tensor_namesConst*
dtype0*!
valueBBdense_4/bias*
_output_shapes
:
k
"save/RestoreV2_24/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_24	RestoreV2
save/Constsave/RestoreV2_24/tensor_names"save/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
¨
save/Assign_24Assigndense_4/biassave/RestoreV2_24*
validate_shape(*
_class
loc:@dense_4/bias*
use_locking(*
T0*
_output_shapes
:

u
save/RestoreV2_25/tensor_namesConst*
dtype0*#
valueBBdense_4/kernel*
_output_shapes
:
k
"save/RestoreV2_25/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_25	RestoreV2
save/Constsave/RestoreV2_25/tensor_names"save/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:
ą
save/Assign_25Assigndense_4/kernelsave/RestoreV2_25*
validate_shape(*!
_class
loc:@dense_4/kernel*
use_locking(*
T0*
_output_shapes
:	

s
save/RestoreV2_26/tensor_namesConst*
dtype0*!
valueBBdense_5/bias*
_output_shapes
:
k
"save/RestoreV2_26/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_26	RestoreV2
save/Constsave/RestoreV2_26/tensor_names"save/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:
¨
save/Assign_26Assigndense_5/biassave/RestoreV2_26*
validate_shape(*
_class
loc:@dense_5/bias*
use_locking(*
T0*
_output_shapes
:

u
save/RestoreV2_27/tensor_namesConst*
dtype0*#
valueBBdense_5/kernel*
_output_shapes
:
k
"save/RestoreV2_27/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_27	RestoreV2
save/Constsave/RestoreV2_27/tensor_names"save/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
ą
save/Assign_27Assigndense_5/kernelsave/RestoreV2_27*
validate_shape(*!
_class
loc:@dense_5/kernel*
use_locking(*
T0*
_output_shapes
:	

s
save/RestoreV2_28/tensor_namesConst*
dtype0*!
valueBBdense_6/bias*
_output_shapes
:
k
"save/RestoreV2_28/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_28	RestoreV2
save/Constsave/RestoreV2_28/tensor_names"save/RestoreV2_28/shape_and_slices*
dtypes
2*
_output_shapes
:
¨
save/Assign_28Assigndense_6/biassave/RestoreV2_28*
validate_shape(*
_class
loc:@dense_6/bias*
use_locking(*
T0*
_output_shapes
:

u
save/RestoreV2_29/tensor_namesConst*
dtype0*#
valueBBdense_6/kernel*
_output_shapes
:
k
"save/RestoreV2_29/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_29	RestoreV2
save/Constsave/RestoreV2_29/tensor_names"save/RestoreV2_29/shape_and_slices*
dtypes
2*
_output_shapes
:
ą
save/Assign_29Assigndense_6/kernelsave/RestoreV2_29*
validate_shape(*!
_class
loc:@dense_6/kernel*
use_locking(*
T0*
_output_shapes
:	

s
save/RestoreV2_30/tensor_namesConst*
dtype0*!
valueBBdense_7/bias*
_output_shapes
:
k
"save/RestoreV2_30/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_30	RestoreV2
save/Constsave/RestoreV2_30/tensor_names"save/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
¨
save/Assign_30Assigndense_7/biassave/RestoreV2_30*
validate_shape(*
_class
loc:@dense_7/bias*
use_locking(*
T0*
_output_shapes
:

u
save/RestoreV2_31/tensor_namesConst*
dtype0*#
valueBBdense_7/kernel*
_output_shapes
:
k
"save/RestoreV2_31/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_31	RestoreV2
save/Constsave/RestoreV2_31/tensor_names"save/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
ą
save/Assign_31Assigndense_7/kernelsave/RestoreV2_31*
validate_shape(*!
_class
loc:@dense_7/kernel*
use_locking(*
T0*
_output_shapes
:	

r
save/RestoreV2_32/tensor_namesConst*
dtype0* 
valueBBglobal_step*
_output_shapes
:
k
"save/RestoreV2_32/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save/RestoreV2_32	RestoreV2
save/Constsave/RestoreV2_32/tensor_names"save/RestoreV2_32/shape_and_slices*
dtypes
2	*
_output_shapes
:
˘
save/Assign_32Assignglobal_stepsave/RestoreV2_32*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0	*
_output_shapes
: 
ż
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_2^save/Assign_3^save/Assign_4^save/Assign_5^save/Assign_6^save/Assign_7^save/Assign_8^save/Assign_9^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_30^save/Assign_31^save/Assign_32
-
save/restore_allNoOp^save/restore_shard

initNoOp

init_all_tablesNoOp

init_1NoOp
4

group_depsNoOp^init^init_all_tables^init_1
R
save_1/ConstConst*
dtype0*
valueB Bmodel*
_output_shapes
: 

save_1/StringJoin/inputs_1Const*
dtype0*<
value3B1 B+_temp_826df885217741ea823e043ddf8403bf/part*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
S
save_1/num_shardsConst*
dtype0*
value	B :*
_output_shapes
: 
m
save_1/ShardedFilename/shardConst"/device:CPU:0*
dtype0*
value	B : *
_output_shapes
: 

save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards"/device:CPU:0*
_output_shapes
: 
č
save_1/SaveV2/tensor_namesConst"/device:CPU:0*
dtype0*
valueBý!Bconv2d/biasBconv2d/kernelBconv2d_1/biasBconv2d_1/kernelBconv2d_2/biasBconv2d_2/kernelBconv2d_3/biasBconv2d_3/kernelBconv2d_4/biasBconv2d_4/kernelBconv2d_5/biasBconv2d_5/kernelBconv2d_6/biasBconv2d_6/kernelBconv2d_7/biasBconv2d_7/kernelB
dense/biasBdense/kernelBdense_1/biasBdense_1/kernelBdense_2/biasBdense_2/kernelBdense_3/biasBdense_3/kernelBdense_4/biasBdense_4/kernelBdense_5/biasBdense_5/kernelBdense_6/biasBdense_6/kernelBdense_7/biasBdense_7/kernelBglobal_step*
_output_shapes
:!
ś
save_1/SaveV2/shape_and_slicesConst"/device:CPU:0*
dtype0*U
valueLBJ!B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
_output_shapes
:!
 
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesconv2d/biasconv2d/kernelconv2d_1/biasconv2d_1/kernelconv2d_2/biasconv2d_2/kernelconv2d_3/biasconv2d_3/kernelconv2d_4/biasconv2d_4/kernelconv2d_5/biasconv2d_5/kernelconv2d_6/biasconv2d_6/kernelconv2d_7/biasconv2d_7/kernel
dense/biasdense/kerneldense_1/biasdense_1/kerneldense_2/biasdense_2/kerneldense_3/biasdense_3/kerneldense_4/biasdense_4/kerneldense_5/biasdense_5/kerneldense_6/biasdense_6/kerneldense_7/biasdense_7/kernelglobal_step"/device:CPU:0*/
dtypes%
#2!	
¨
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2"/device:CPU:0*)
_class
loc:@save_1/ShardedFilename*
T0*
_output_shapes
: 
˛
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency"/device:CPU:0*
N*

axis *
T0*
_output_shapes
:

save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const"/device:CPU:0*
delete_old_dirs(

save_1/IdentityIdentitysave_1/Const^save_1/control_dependency^save_1/MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 
q
save_1/RestoreV2/tensor_namesConst*
dtype0* 
valueBBconv2d/bias*
_output_shapes
:
j
!save_1/RestoreV2/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*
dtypes
2*
_output_shapes
:
¤
save_1/AssignAssignconv2d/biassave_1/RestoreV2*
validate_shape(*
_class
loc:@conv2d/bias*
use_locking(*
T0*
_output_shapes
: 
u
save_1/RestoreV2_1/tensor_namesConst*
dtype0*"
valueBBconv2d/kernel*
_output_shapes
:
l
#save_1/RestoreV2_1/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save_1/RestoreV2_1	RestoreV2save_1/Constsave_1/RestoreV2_1/tensor_names#save_1/RestoreV2_1/shape_and_slices*
dtypes
2*
_output_shapes
:
¸
save_1/Assign_1Assignconv2d/kernelsave_1/RestoreV2_1*
validate_shape(* 
_class
loc:@conv2d/kernel*
use_locking(*
T0*&
_output_shapes
: 
u
save_1/RestoreV2_2/tensor_namesConst*
dtype0*"
valueBBconv2d_1/bias*
_output_shapes
:
l
#save_1/RestoreV2_2/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save_1/RestoreV2_2	RestoreV2save_1/Constsave_1/RestoreV2_2/tensor_names#save_1/RestoreV2_2/shape_and_slices*
dtypes
2*
_output_shapes
:
Ź
save_1/Assign_2Assignconv2d_1/biassave_1/RestoreV2_2*
validate_shape(* 
_class
loc:@conv2d_1/bias*
use_locking(*
T0*
_output_shapes
: 
w
save_1/RestoreV2_3/tensor_namesConst*
dtype0*$
valueBBconv2d_1/kernel*
_output_shapes
:
l
#save_1/RestoreV2_3/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save_1/RestoreV2_3	RestoreV2save_1/Constsave_1/RestoreV2_3/tensor_names#save_1/RestoreV2_3/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_1/Assign_3Assignconv2d_1/kernelsave_1/RestoreV2_3*
validate_shape(*"
_class
loc:@conv2d_1/kernel*
use_locking(*
T0*&
_output_shapes
: 
u
save_1/RestoreV2_4/tensor_namesConst*
dtype0*"
valueBBconv2d_2/bias*
_output_shapes
:
l
#save_1/RestoreV2_4/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save_1/RestoreV2_4	RestoreV2save_1/Constsave_1/RestoreV2_4/tensor_names#save_1/RestoreV2_4/shape_and_slices*
dtypes
2*
_output_shapes
:
Ź
save_1/Assign_4Assignconv2d_2/biassave_1/RestoreV2_4*
validate_shape(* 
_class
loc:@conv2d_2/bias*
use_locking(*
T0*
_output_shapes
: 
w
save_1/RestoreV2_5/tensor_namesConst*
dtype0*$
valueBBconv2d_2/kernel*
_output_shapes
:
l
#save_1/RestoreV2_5/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save_1/RestoreV2_5	RestoreV2save_1/Constsave_1/RestoreV2_5/tensor_names#save_1/RestoreV2_5/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_1/Assign_5Assignconv2d_2/kernelsave_1/RestoreV2_5*
validate_shape(*"
_class
loc:@conv2d_2/kernel*
use_locking(*
T0*&
_output_shapes
: 
u
save_1/RestoreV2_6/tensor_namesConst*
dtype0*"
valueBBconv2d_3/bias*
_output_shapes
:
l
#save_1/RestoreV2_6/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save_1/RestoreV2_6	RestoreV2save_1/Constsave_1/RestoreV2_6/tensor_names#save_1/RestoreV2_6/shape_and_slices*
dtypes
2*
_output_shapes
:
Ź
save_1/Assign_6Assignconv2d_3/biassave_1/RestoreV2_6*
validate_shape(* 
_class
loc:@conv2d_3/bias*
use_locking(*
T0*
_output_shapes
: 
w
save_1/RestoreV2_7/tensor_namesConst*
dtype0*$
valueBBconv2d_3/kernel*
_output_shapes
:
l
#save_1/RestoreV2_7/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save_1/RestoreV2_7	RestoreV2save_1/Constsave_1/RestoreV2_7/tensor_names#save_1/RestoreV2_7/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_1/Assign_7Assignconv2d_3/kernelsave_1/RestoreV2_7*
validate_shape(*"
_class
loc:@conv2d_3/kernel*
use_locking(*
T0*&
_output_shapes
: 
u
save_1/RestoreV2_8/tensor_namesConst*
dtype0*"
valueBBconv2d_4/bias*
_output_shapes
:
l
#save_1/RestoreV2_8/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save_1/RestoreV2_8	RestoreV2save_1/Constsave_1/RestoreV2_8/tensor_names#save_1/RestoreV2_8/shape_and_slices*
dtypes
2*
_output_shapes
:
Ź
save_1/Assign_8Assignconv2d_4/biassave_1/RestoreV2_8*
validate_shape(* 
_class
loc:@conv2d_4/bias*
use_locking(*
T0*
_output_shapes
:@
w
save_1/RestoreV2_9/tensor_namesConst*
dtype0*$
valueBBconv2d_4/kernel*
_output_shapes
:
l
#save_1/RestoreV2_9/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:

save_1/RestoreV2_9	RestoreV2save_1/Constsave_1/RestoreV2_9/tensor_names#save_1/RestoreV2_9/shape_and_slices*
dtypes
2*
_output_shapes
:
ź
save_1/Assign_9Assignconv2d_4/kernelsave_1/RestoreV2_9*
validate_shape(*"
_class
loc:@conv2d_4/kernel*
use_locking(*
T0*&
_output_shapes
: @
v
 save_1/RestoreV2_10/tensor_namesConst*
dtype0*"
valueBBconv2d_5/bias*
_output_shapes
:
m
$save_1/RestoreV2_10/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ą
save_1/RestoreV2_10	RestoreV2save_1/Const save_1/RestoreV2_10/tensor_names$save_1/RestoreV2_10/shape_and_slices*
dtypes
2*
_output_shapes
:
Ž
save_1/Assign_10Assignconv2d_5/biassave_1/RestoreV2_10*
validate_shape(* 
_class
loc:@conv2d_5/bias*
use_locking(*
T0*
_output_shapes
:@
x
 save_1/RestoreV2_11/tensor_namesConst*
dtype0*$
valueBBconv2d_5/kernel*
_output_shapes
:
m
$save_1/RestoreV2_11/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ą
save_1/RestoreV2_11	RestoreV2save_1/Const save_1/RestoreV2_11/tensor_names$save_1/RestoreV2_11/shape_and_slices*
dtypes
2*
_output_shapes
:
ž
save_1/Assign_11Assignconv2d_5/kernelsave_1/RestoreV2_11*
validate_shape(*"
_class
loc:@conv2d_5/kernel*
use_locking(*
T0*&
_output_shapes
: @
v
 save_1/RestoreV2_12/tensor_namesConst*
dtype0*"
valueBBconv2d_6/bias*
_output_shapes
:
m
$save_1/RestoreV2_12/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ą
save_1/RestoreV2_12	RestoreV2save_1/Const save_1/RestoreV2_12/tensor_names$save_1/RestoreV2_12/shape_and_slices*
dtypes
2*
_output_shapes
:
Ž
save_1/Assign_12Assignconv2d_6/biassave_1/RestoreV2_12*
validate_shape(* 
_class
loc:@conv2d_6/bias*
use_locking(*
T0*
_output_shapes
:@
x
 save_1/RestoreV2_13/tensor_namesConst*
dtype0*$
valueBBconv2d_6/kernel*
_output_shapes
:
m
$save_1/RestoreV2_13/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ą
save_1/RestoreV2_13	RestoreV2save_1/Const save_1/RestoreV2_13/tensor_names$save_1/RestoreV2_13/shape_and_slices*
dtypes
2*
_output_shapes
:
ž
save_1/Assign_13Assignconv2d_6/kernelsave_1/RestoreV2_13*
validate_shape(*"
_class
loc:@conv2d_6/kernel*
use_locking(*
T0*&
_output_shapes
: @
v
 save_1/RestoreV2_14/tensor_namesConst*
dtype0*"
valueBBconv2d_7/bias*
_output_shapes
:
m
$save_1/RestoreV2_14/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ą
save_1/RestoreV2_14	RestoreV2save_1/Const save_1/RestoreV2_14/tensor_names$save_1/RestoreV2_14/shape_and_slices*
dtypes
2*
_output_shapes
:
Ž
save_1/Assign_14Assignconv2d_7/biassave_1/RestoreV2_14*
validate_shape(* 
_class
loc:@conv2d_7/bias*
use_locking(*
T0*
_output_shapes
:@
x
 save_1/RestoreV2_15/tensor_namesConst*
dtype0*$
valueBBconv2d_7/kernel*
_output_shapes
:
m
$save_1/RestoreV2_15/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ą
save_1/RestoreV2_15	RestoreV2save_1/Const save_1/RestoreV2_15/tensor_names$save_1/RestoreV2_15/shape_and_slices*
dtypes
2*
_output_shapes
:
ž
save_1/Assign_15Assignconv2d_7/kernelsave_1/RestoreV2_15*
validate_shape(*"
_class
loc:@conv2d_7/kernel*
use_locking(*
T0*&
_output_shapes
: @
s
 save_1/RestoreV2_16/tensor_namesConst*
dtype0*
valueBB
dense/bias*
_output_shapes
:
m
$save_1/RestoreV2_16/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ą
save_1/RestoreV2_16	RestoreV2save_1/Const save_1/RestoreV2_16/tensor_names$save_1/RestoreV2_16/shape_and_slices*
dtypes
2*
_output_shapes
:
Š
save_1/Assign_16Assign
dense/biassave_1/RestoreV2_16*
validate_shape(*
_class
loc:@dense/bias*
use_locking(*
T0*
_output_shapes	
:
u
 save_1/RestoreV2_17/tensor_namesConst*
dtype0*!
valueBBdense/kernel*
_output_shapes
:
m
$save_1/RestoreV2_17/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ą
save_1/RestoreV2_17	RestoreV2save_1/Const save_1/RestoreV2_17/tensor_names$save_1/RestoreV2_17/shape_and_slices*
dtypes
2*
_output_shapes
:
˛
save_1/Assign_17Assigndense/kernelsave_1/RestoreV2_17*
validate_shape(*
_class
loc:@dense/kernel*
use_locking(*
T0* 
_output_shapes
:
Ŕ
u
 save_1/RestoreV2_18/tensor_namesConst*
dtype0*!
valueBBdense_1/bias*
_output_shapes
:
m
$save_1/RestoreV2_18/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ą
save_1/RestoreV2_18	RestoreV2save_1/Const save_1/RestoreV2_18/tensor_names$save_1/RestoreV2_18/shape_and_slices*
dtypes
2*
_output_shapes
:
­
save_1/Assign_18Assigndense_1/biassave_1/RestoreV2_18*
validate_shape(*
_class
loc:@dense_1/bias*
use_locking(*
T0*
_output_shapes	
:
w
 save_1/RestoreV2_19/tensor_namesConst*
dtype0*#
valueBBdense_1/kernel*
_output_shapes
:
m
$save_1/RestoreV2_19/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ą
save_1/RestoreV2_19	RestoreV2save_1/Const save_1/RestoreV2_19/tensor_names$save_1/RestoreV2_19/shape_and_slices*
dtypes
2*
_output_shapes
:
ś
save_1/Assign_19Assigndense_1/kernelsave_1/RestoreV2_19*
validate_shape(*!
_class
loc:@dense_1/kernel*
use_locking(*
T0* 
_output_shapes
:
Ŕ
u
 save_1/RestoreV2_20/tensor_namesConst*
dtype0*!
valueBBdense_2/bias*
_output_shapes
:
m
$save_1/RestoreV2_20/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ą
save_1/RestoreV2_20	RestoreV2save_1/Const save_1/RestoreV2_20/tensor_names$save_1/RestoreV2_20/shape_and_slices*
dtypes
2*
_output_shapes
:
­
save_1/Assign_20Assigndense_2/biassave_1/RestoreV2_20*
validate_shape(*
_class
loc:@dense_2/bias*
use_locking(*
T0*
_output_shapes	
:
w
 save_1/RestoreV2_21/tensor_namesConst*
dtype0*#
valueBBdense_2/kernel*
_output_shapes
:
m
$save_1/RestoreV2_21/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ą
save_1/RestoreV2_21	RestoreV2save_1/Const save_1/RestoreV2_21/tensor_names$save_1/RestoreV2_21/shape_and_slices*
dtypes
2*
_output_shapes
:
ś
save_1/Assign_21Assigndense_2/kernelsave_1/RestoreV2_21*
validate_shape(*!
_class
loc:@dense_2/kernel*
use_locking(*
T0* 
_output_shapes
:
Ŕ
u
 save_1/RestoreV2_22/tensor_namesConst*
dtype0*!
valueBBdense_3/bias*
_output_shapes
:
m
$save_1/RestoreV2_22/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ą
save_1/RestoreV2_22	RestoreV2save_1/Const save_1/RestoreV2_22/tensor_names$save_1/RestoreV2_22/shape_and_slices*
dtypes
2*
_output_shapes
:
­
save_1/Assign_22Assigndense_3/biassave_1/RestoreV2_22*
validate_shape(*
_class
loc:@dense_3/bias*
use_locking(*
T0*
_output_shapes	
:
w
 save_1/RestoreV2_23/tensor_namesConst*
dtype0*#
valueBBdense_3/kernel*
_output_shapes
:
m
$save_1/RestoreV2_23/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ą
save_1/RestoreV2_23	RestoreV2save_1/Const save_1/RestoreV2_23/tensor_names$save_1/RestoreV2_23/shape_and_slices*
dtypes
2*
_output_shapes
:
ś
save_1/Assign_23Assigndense_3/kernelsave_1/RestoreV2_23*
validate_shape(*!
_class
loc:@dense_3/kernel*
use_locking(*
T0* 
_output_shapes
:
Ŕ
u
 save_1/RestoreV2_24/tensor_namesConst*
dtype0*!
valueBBdense_4/bias*
_output_shapes
:
m
$save_1/RestoreV2_24/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ą
save_1/RestoreV2_24	RestoreV2save_1/Const save_1/RestoreV2_24/tensor_names$save_1/RestoreV2_24/shape_and_slices*
dtypes
2*
_output_shapes
:
Ź
save_1/Assign_24Assigndense_4/biassave_1/RestoreV2_24*
validate_shape(*
_class
loc:@dense_4/bias*
use_locking(*
T0*
_output_shapes
:

w
 save_1/RestoreV2_25/tensor_namesConst*
dtype0*#
valueBBdense_4/kernel*
_output_shapes
:
m
$save_1/RestoreV2_25/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ą
save_1/RestoreV2_25	RestoreV2save_1/Const save_1/RestoreV2_25/tensor_names$save_1/RestoreV2_25/shape_and_slices*
dtypes
2*
_output_shapes
:
ľ
save_1/Assign_25Assigndense_4/kernelsave_1/RestoreV2_25*
validate_shape(*!
_class
loc:@dense_4/kernel*
use_locking(*
T0*
_output_shapes
:	

u
 save_1/RestoreV2_26/tensor_namesConst*
dtype0*!
valueBBdense_5/bias*
_output_shapes
:
m
$save_1/RestoreV2_26/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ą
save_1/RestoreV2_26	RestoreV2save_1/Const save_1/RestoreV2_26/tensor_names$save_1/RestoreV2_26/shape_and_slices*
dtypes
2*
_output_shapes
:
Ź
save_1/Assign_26Assigndense_5/biassave_1/RestoreV2_26*
validate_shape(*
_class
loc:@dense_5/bias*
use_locking(*
T0*
_output_shapes
:

w
 save_1/RestoreV2_27/tensor_namesConst*
dtype0*#
valueBBdense_5/kernel*
_output_shapes
:
m
$save_1/RestoreV2_27/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ą
save_1/RestoreV2_27	RestoreV2save_1/Const save_1/RestoreV2_27/tensor_names$save_1/RestoreV2_27/shape_and_slices*
dtypes
2*
_output_shapes
:
ľ
save_1/Assign_27Assigndense_5/kernelsave_1/RestoreV2_27*
validate_shape(*!
_class
loc:@dense_5/kernel*
use_locking(*
T0*
_output_shapes
:	

u
 save_1/RestoreV2_28/tensor_namesConst*
dtype0*!
valueBBdense_6/bias*
_output_shapes
:
m
$save_1/RestoreV2_28/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ą
save_1/RestoreV2_28	RestoreV2save_1/Const save_1/RestoreV2_28/tensor_names$save_1/RestoreV2_28/shape_and_slices*
dtypes
2*
_output_shapes
:
Ź
save_1/Assign_28Assigndense_6/biassave_1/RestoreV2_28*
validate_shape(*
_class
loc:@dense_6/bias*
use_locking(*
T0*
_output_shapes
:

w
 save_1/RestoreV2_29/tensor_namesConst*
dtype0*#
valueBBdense_6/kernel*
_output_shapes
:
m
$save_1/RestoreV2_29/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ą
save_1/RestoreV2_29	RestoreV2save_1/Const save_1/RestoreV2_29/tensor_names$save_1/RestoreV2_29/shape_and_slices*
dtypes
2*
_output_shapes
:
ľ
save_1/Assign_29Assigndense_6/kernelsave_1/RestoreV2_29*
validate_shape(*!
_class
loc:@dense_6/kernel*
use_locking(*
T0*
_output_shapes
:	

u
 save_1/RestoreV2_30/tensor_namesConst*
dtype0*!
valueBBdense_7/bias*
_output_shapes
:
m
$save_1/RestoreV2_30/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ą
save_1/RestoreV2_30	RestoreV2save_1/Const save_1/RestoreV2_30/tensor_names$save_1/RestoreV2_30/shape_and_slices*
dtypes
2*
_output_shapes
:
Ź
save_1/Assign_30Assigndense_7/biassave_1/RestoreV2_30*
validate_shape(*
_class
loc:@dense_7/bias*
use_locking(*
T0*
_output_shapes
:

w
 save_1/RestoreV2_31/tensor_namesConst*
dtype0*#
valueBBdense_7/kernel*
_output_shapes
:
m
$save_1/RestoreV2_31/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ą
save_1/RestoreV2_31	RestoreV2save_1/Const save_1/RestoreV2_31/tensor_names$save_1/RestoreV2_31/shape_and_slices*
dtypes
2*
_output_shapes
:
ľ
save_1/Assign_31Assigndense_7/kernelsave_1/RestoreV2_31*
validate_shape(*!
_class
loc:@dense_7/kernel*
use_locking(*
T0*
_output_shapes
:	

t
 save_1/RestoreV2_32/tensor_namesConst*
dtype0* 
valueBBglobal_step*
_output_shapes
:
m
$save_1/RestoreV2_32/shape_and_slicesConst*
dtype0*
valueB
B *
_output_shapes
:
Ą
save_1/RestoreV2_32	RestoreV2save_1/Const save_1/RestoreV2_32/tensor_names$save_1/RestoreV2_32/shape_and_slices*
dtypes
2	*
_output_shapes
:
Ś
save_1/Assign_32Assignglobal_stepsave_1/RestoreV2_32*
validate_shape(*
_class
loc:@global_step*
use_locking(*
T0	*
_output_shapes
: 

save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_2^save_1/Assign_3^save_1/Assign_4^save_1/Assign_5^save_1/Assign_6^save_1/Assign_7^save_1/Assign_8^save_1/Assign_9^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32
1
save_1/restore_allNoOp^save_1/restore_shard"B
save_1/Const:0save_1/Identity:0save_1/restore_all (5 @F8"Ű
	variablesÍĘ
X
global_step:0global_step/Assignglobal_step/read:02global_step/Initializer/zeros:0
i
conv2d/kernel:0conv2d/kernel/Assignconv2d/kernel/read:02*conv2d/kernel/Initializer/random_uniform:0
X
conv2d/bias:0conv2d/bias/Assignconv2d/bias/read:02conv2d/bias/Initializer/zeros:0
q
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02,conv2d_1/kernel/Initializer/random_uniform:0
`
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:02!conv2d_1/bias/Initializer/zeros:0
q
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:02,conv2d_2/kernel/Initializer/random_uniform:0
`
conv2d_2/bias:0conv2d_2/bias/Assignconv2d_2/bias/read:02!conv2d_2/bias/Initializer/zeros:0
q
conv2d_3/kernel:0conv2d_3/kernel/Assignconv2d_3/kernel/read:02,conv2d_3/kernel/Initializer/random_uniform:0
`
conv2d_3/bias:0conv2d_3/bias/Assignconv2d_3/bias/read:02!conv2d_3/bias/Initializer/zeros:0
q
conv2d_4/kernel:0conv2d_4/kernel/Assignconv2d_4/kernel/read:02,conv2d_4/kernel/Initializer/random_uniform:0
`
conv2d_4/bias:0conv2d_4/bias/Assignconv2d_4/bias/read:02!conv2d_4/bias/Initializer/zeros:0
q
conv2d_5/kernel:0conv2d_5/kernel/Assignconv2d_5/kernel/read:02,conv2d_5/kernel/Initializer/random_uniform:0
`
conv2d_5/bias:0conv2d_5/bias/Assignconv2d_5/bias/read:02!conv2d_5/bias/Initializer/zeros:0
q
conv2d_6/kernel:0conv2d_6/kernel/Assignconv2d_6/kernel/read:02,conv2d_6/kernel/Initializer/random_uniform:0
`
conv2d_6/bias:0conv2d_6/bias/Assignconv2d_6/bias/read:02!conv2d_6/bias/Initializer/zeros:0
q
conv2d_7/kernel:0conv2d_7/kernel/Assignconv2d_7/kernel/read:02,conv2d_7/kernel/Initializer/random_uniform:0
`
conv2d_7/bias:0conv2d_7/bias/Assignconv2d_7/bias/read:02!conv2d_7/bias/Initializer/zeros:0
e
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:0
T
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:0
m
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:0
\
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:0
m
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:02+dense_2/kernel/Initializer/random_uniform:0
\
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:02 dense_2/bias/Initializer/zeros:0
m
dense_3/kernel:0dense_3/kernel/Assigndense_3/kernel/read:02+dense_3/kernel/Initializer/random_uniform:0
\
dense_3/bias:0dense_3/bias/Assigndense_3/bias/read:02 dense_3/bias/Initializer/zeros:0
m
dense_4/kernel:0dense_4/kernel/Assigndense_4/kernel/read:02+dense_4/kernel/Initializer/random_uniform:0
\
dense_4/bias:0dense_4/bias/Assigndense_4/bias/read:02 dense_4/bias/Initializer/zeros:0
m
dense_5/kernel:0dense_5/kernel/Assigndense_5/kernel/read:02+dense_5/kernel/Initializer/random_uniform:0
\
dense_5/bias:0dense_5/bias/Assigndense_5/bias/read:02 dense_5/bias/Initializer/zeros:0
m
dense_6/kernel:0dense_6/kernel/Assigndense_6/kernel/read:02+dense_6/kernel/Initializer/random_uniform:0
\
dense_6/bias:0dense_6/bias/Assigndense_6/bias/read:02 dense_6/bias/Initializer/zeros:0
m
dense_7/kernel:0dense_7/kernel/Assigndense_7/kernel/read:02+dense_7/kernel/Initializer/random_uniform:0
\
dense_7/bias:0dense_7/bias/Assigndense_7/bias/read:02 dense_7/bias/Initializer/zeros:0" 
global_step

global_step:0" 
legacy_init_op


group_deps"
trainable_variablesóđ
i
conv2d/kernel:0conv2d/kernel/Assignconv2d/kernel/read:02*conv2d/kernel/Initializer/random_uniform:0
X
conv2d/bias:0conv2d/bias/Assignconv2d/bias/read:02conv2d/bias/Initializer/zeros:0
q
conv2d_1/kernel:0conv2d_1/kernel/Assignconv2d_1/kernel/read:02,conv2d_1/kernel/Initializer/random_uniform:0
`
conv2d_1/bias:0conv2d_1/bias/Assignconv2d_1/bias/read:02!conv2d_1/bias/Initializer/zeros:0
q
conv2d_2/kernel:0conv2d_2/kernel/Assignconv2d_2/kernel/read:02,conv2d_2/kernel/Initializer/random_uniform:0
`
conv2d_2/bias:0conv2d_2/bias/Assignconv2d_2/bias/read:02!conv2d_2/bias/Initializer/zeros:0
q
conv2d_3/kernel:0conv2d_3/kernel/Assignconv2d_3/kernel/read:02,conv2d_3/kernel/Initializer/random_uniform:0
`
conv2d_3/bias:0conv2d_3/bias/Assignconv2d_3/bias/read:02!conv2d_3/bias/Initializer/zeros:0
q
conv2d_4/kernel:0conv2d_4/kernel/Assignconv2d_4/kernel/read:02,conv2d_4/kernel/Initializer/random_uniform:0
`
conv2d_4/bias:0conv2d_4/bias/Assignconv2d_4/bias/read:02!conv2d_4/bias/Initializer/zeros:0
q
conv2d_5/kernel:0conv2d_5/kernel/Assignconv2d_5/kernel/read:02,conv2d_5/kernel/Initializer/random_uniform:0
`
conv2d_5/bias:0conv2d_5/bias/Assignconv2d_5/bias/read:02!conv2d_5/bias/Initializer/zeros:0
q
conv2d_6/kernel:0conv2d_6/kernel/Assignconv2d_6/kernel/read:02,conv2d_6/kernel/Initializer/random_uniform:0
`
conv2d_6/bias:0conv2d_6/bias/Assignconv2d_6/bias/read:02!conv2d_6/bias/Initializer/zeros:0
q
conv2d_7/kernel:0conv2d_7/kernel/Assignconv2d_7/kernel/read:02,conv2d_7/kernel/Initializer/random_uniform:0
`
conv2d_7/bias:0conv2d_7/bias/Assignconv2d_7/bias/read:02!conv2d_7/bias/Initializer/zeros:0
e
dense/kernel:0dense/kernel/Assigndense/kernel/read:02)dense/kernel/Initializer/random_uniform:0
T
dense/bias:0dense/bias/Assigndense/bias/read:02dense/bias/Initializer/zeros:0
m
dense_1/kernel:0dense_1/kernel/Assigndense_1/kernel/read:02+dense_1/kernel/Initializer/random_uniform:0
\
dense_1/bias:0dense_1/bias/Assigndense_1/bias/read:02 dense_1/bias/Initializer/zeros:0
m
dense_2/kernel:0dense_2/kernel/Assigndense_2/kernel/read:02+dense_2/kernel/Initializer/random_uniform:0
\
dense_2/bias:0dense_2/bias/Assigndense_2/bias/read:02 dense_2/bias/Initializer/zeros:0
m
dense_3/kernel:0dense_3/kernel/Assigndense_3/kernel/read:02+dense_3/kernel/Initializer/random_uniform:0
\
dense_3/bias:0dense_3/bias/Assigndense_3/bias/read:02 dense_3/bias/Initializer/zeros:0
m
dense_4/kernel:0dense_4/kernel/Assigndense_4/kernel/read:02+dense_4/kernel/Initializer/random_uniform:0
\
dense_4/bias:0dense_4/bias/Assigndense_4/bias/read:02 dense_4/bias/Initializer/zeros:0
m
dense_5/kernel:0dense_5/kernel/Assigndense_5/kernel/read:02+dense_5/kernel/Initializer/random_uniform:0
\
dense_5/bias:0dense_5/bias/Assigndense_5/bias/read:02 dense_5/bias/Initializer/zeros:0
m
dense_6/kernel:0dense_6/kernel/Assigndense_6/kernel/read:02+dense_6/kernel/Initializer/random_uniform:0
\
dense_6/bias:0dense_6/bias/Assigndense_6/bias/read:02 dense_6/bias/Initializer/zeros:0
m
dense_7/kernel:0dense_7/kernel/Assigndense_7/kernel/read:02+dense_7/kernel/Initializer/random_uniform:0
\
dense_7/bias:0dense_7/bias/Assigndense_7/bias/read:02 dense_7/bias/Initializer/zeros:0