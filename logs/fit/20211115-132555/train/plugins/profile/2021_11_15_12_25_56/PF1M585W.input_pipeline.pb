$	?DZ???????]????'???????!?ڊ?e???$	Uf??m?@@0`Bo@l?Wb?V@!?Ё?I-@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??MbX??????Mb??A??+e???Y???N@??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ڊ?e???j+?????A?鷯??YK?=?U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&q???h ??
h"lxz??A?^)???YX9??v???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&d;?O????Ș?????A䃞ͪ???YˡE?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Zd;?O????ǘ?????Aio???T??YD?l?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?`TR'???R'??????A??????Y{?G?z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ʡE???????QI???A䃞ͪ???Y??&???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??S㥛??EGr????A"?uq??Yn????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??|гY??????????A??	h"l??Y??&???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	-C??6??2U0*???A?%䃞???Y??@??ǘ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
-C??6??t??????A?6?[ ??Y'???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&!?lV}????S㥛??A?A`??"??Y?N@aÓ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&B`??"???L7?A`???A??&S??Y??ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?D?????I??&??AjM??S??Y?v??/??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?^)?????"??~j??A?Q???Y??ܵ?|??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?d?`TR??A?c?]K??A?ܵ?|???Y?Q?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?lV}???!?rh????A???ׁs??YX9??v???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??g?????5?;N???A??ǘ????Y??ͪ?Ֆ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ڊ?e????&?W??A?G?z??Y??0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?=yX??????_vO??A=
ףp=??Y??_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&'???????ˡE?????A???QI???Yw-!?l??*	?????x?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??C?l??!???%?A@)鷯???1{?k~f?@:Preprocessing2F
Iterator::Model???JY???!??/???A@)c?ZB>???1?}֡rt3@:Preprocessing2U
Iterator::Model::ParallelMapV2????H??!F??|s0@)????H??1F??|s0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipV}??b??!h3P@)H?}8g??1X??*&@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate1?*?Թ?!???Z,@)	?c???1TK?o@B@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice_)?Ǻ??!;?kE??@)_)?Ǻ??1;?kE??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??^??!N??s??2@)?5?;Nѡ?1??*?[@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*????<,??!d?????@)????<,??1d?????@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 49.7% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9????@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	7U????????`,?<??ˡE?????!?&?W??	!       "	!       *	!       2$	?????????"y???????QI???!?%䃞???:	!       B	!       J$	"e3?')??xu7͡??X9??v???!???N@??R	!       Z$	"e3?')??xu7͡??X9??v???!???N@??JCPU_ONLYY????@b 