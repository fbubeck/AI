?($	??g<;G??㫚 U???L7?A`???!鷯???$	;?TΒ_@?n&H'?
@?9?h???!qR
???1@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$4??7???? o?ŏ??A^?I+??Y6<?R???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???(???d;?O????A??{??P??Yŏ1w-!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&L?
F%u??ףp=
???A???9#J??Y?,C????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&鷯????Pk?w???AjM??S??Y??ܵ?|??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&0?'????{??Pk??Au?V??Y?sF????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??/?$???h o???A2w-!???Y?:pΈ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&{?/L?
???|гY???A?	h"lx??Y??ׁsF??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????K??Ӽ????A??????Y?&S???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&+???????X?? ??A2w-!???YK?=?U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	??	h"??!?lV}??A"?uq??Y???S㥛?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
o???T????Q?????A????????Y2U0*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ŏ1w??%u???A?G?z???Y? ?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&M?J???f??a????A????K7??YX9??v???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Pk?w???[????<??AX9??v??Y???߾??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?*??	???k	??g??A?p=
ף??YT㥛? ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?p=
ף??q???h ??A???T????Yc?ZB>???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&x$(~??\ A?c???A??(\????Y??ͪ?Ֆ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?G?z????߾?3??AD????9??Y?W[?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&c?ZB>???m???{???Alxz?,C??Yf??a?֤?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ܵ?|???d?]K???A?(??0??Y?I+???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&L7?A`????A`??"??AJ+???YJ+???*	gffff
?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??(\????!??H?@@)?x?&1??1o0A?<@:Preprocessing2F
Iterator::Model?4?8EG??!??1JD@)&S????1??????7@:Preprocessing2U
Iterator::Model::ParallelMapV2?C?l????!tzj?<0@)?C?l????1tzj?<0@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip鷯????!????M@)Ӽ?ɵ?1?^S?b?#@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??z6???!\?????+@)?sF????1?I@?n@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice???_vO??!?ƶV?]@)???_vO??1?ƶV?]@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap???????!?M????1@)?5?;Nѡ?1V%Z}?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*???S㥛?!???b?@)???S㥛?1???b?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 50.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?s?z[?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	N?`+g??????????A`??"??!?Pk?w???	!       "	!       *	!       2$	??|hw???`~1?ȸ?^?I+??!?	h"lx??:	!       B	!       J$	?=?'wG???k]?????K?=?U??!6<?R???R	!       Z$	?=?'wG???k]?????K?=?U??!6<?R???JCPU_ONLYY?s?z[?@b Y      Y@q???HxA@"?
both?Your program is POTENTIALLY input-bound because 50.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?34.9397% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 