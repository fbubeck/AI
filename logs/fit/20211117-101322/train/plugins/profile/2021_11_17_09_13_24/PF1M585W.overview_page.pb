?($	?V??????R?????g??s???!;pΈ????$	?t#?9@?????:@????Z?@!u??H?)@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?U???????&?W??AD?l?????Y??Ƭ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&jM??S??Ԛ?????A?٬?\m??Y?:pΈ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&vOjM??-???????A??D????Yh??|?5??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&4??7?????	???AP??n???Yh??|?5??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ʡE?????n????A???Mb??YV-???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&o?ŏ1??5^?I??A???镲??Y??ܵ?|??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&sh??|???&S??:??A??~j?t??YB>?٬???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Zd;?O??
h"lxz??A1?Zd??Y???߾??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?uq?????JY?8??A??3????Y?v??/??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	J+?????V?/???A?St$????YS?!?uq??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?HP???"??u????AV????_??Y46<?R??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Y??ڊ???H?}8??A4??7????Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&;pΈ???????x?&??A?H?}8??Y'???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Έ?????w-!?l??A?-????Y?=yX???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&]m???{???A`??"??A?@??ǘ??Y??B?iޡ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?(??0??Tt$?????A???????Y;?O??n??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??&S??]m???{??A??Q???Y????ׁ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ףp=
???}гY????A?D?????Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????K7??ۊ?e????A-??????Y%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???1?????B?i?q??A?HP???Y?v??/??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?g??s????J?4??A?????Y?!??u???*	     |?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??ݓ????!?=E?vA@)?~?:p???1???M"@@:Preprocessing2F
Iterator::ModelQk?w????!?v???SB@)??e?c]??1??aº?5@:Preprocessing2U
Iterator::Model::ParallelMapV2?A?f????!4Z?Q-@)?A?f????14Z?Q-@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip??	h"l??!6?8&?O@)z6?>W??1???t3%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?:pΈ??!????@)?:pΈ??1????@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?}8gD??!>E?v?*@)      ??1?|Ty[?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap`??"????!??#??1@)??ͪ?զ?1z???
?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*S?!?uq??![?h?G@)S?!?uq??1[?h?G@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 52.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9=5'k[@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	T?????????r???J?4??!???x?&??	!       "	!       *	!       2$	%˝.?A???d??''???????!?H?}8??:	!       B	!       J$	?????????#??S?!?uq??!??Ƭ?R	!       Z$	?????????#??S?!?uq??!??Ƭ?JCPU_ONLYY=5'k[@b Y      Y@q???aT@"?
both?Your program is POTENTIALLY input-bound because 52.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?80.0685% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 