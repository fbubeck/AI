?)$	??iĢ???xϥ?j??F????x??!??q????$	R"M,?@qX??s@a??o?@!?3J???:@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?G?z??0*??D??A_?L?J??Yd]?Fx??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&K?46??Έ?????A+??	h??Y???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&T㥛? ??鷯???AH?z?G??Y??ǘ????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&mV}??b??Zd;?O???Aq???h??Y???~?:??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&+?????z6?>W[??A?{??Pk??Y	?^)ˠ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Zd;?O????C??????A?R?!?u??Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??(????F%u???A?lV}????Yz6?>W[??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??7??d???ׁsF???A<?R?!???Y?,C????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??q????V-????A??ʡE???YK?46??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?{??Pk??1?Zd??A?[ A?c??YM?J???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
.???1???xz?,C??AtF??_??Y?:pΈ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???JY???$???~???A??u????Y???JY???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&(??y???ͪ??V??A ?o_???Yz6?>W??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&/?$?????:M??A??|?5^??Y?lV}????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?C?l??????K7?A??A?9#J{???Y*:??H??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?o_???}гY????A?+e?X??Yޓ??ZӬ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&F%u?????|гY??AK?46??Y??H.?!??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&333333??n????A??????Y/?$???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??Q???>?٬?\??A`??"????Y??JY?8??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&j?t???]?C?????A???????Y?^)?Ǫ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&F????x???Ǻ????A&S????Y8??d?`??*	???????@2F
Iterator::ModelΈ?????!????~?G@)f?c]?F??1cx???B@:Preprocessing2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?Q?|??!?#/?>@)??	h"l??1?o&??=@:Preprocessing2U
Iterator::Model::ParallelMapV2????!y??w;n$@)????1y??w;n$@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???h o??!>I{?J@)????????1W?đ?@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlice?^)???!:Q??Tk@)?^)???1:Q??Tk@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate?;Nё\??!?+D???#@)?=yX???1???<,@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap?ʡE????!X?a??*@)?;Nё\??1Q?v???
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?St$????!??????)?St$????1??????:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is MODERATELY input-bound because 5.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.no*high2t53.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.9?TN`?I@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	U?I??????{?Y????Ǻ????!V-????	!       "	!       *	!       2$	??)cԋ??'??T????&S????!??|?5^??:	!       B	!       J$	,.M?i,??Ѡ?f?(?????~?:??!d]?Fx??R	!       Z$	,.M?i,??Ѡ?f?(?????~?:??!d]?Fx??JCPU_ONLYY?TN`?I@b Y      Y@q???YdC@"?	
both?Your program is MODERATELY input-bound because 5.8% of the total step time sampled is waiting for input. Therefore, you would need to reduce both the input time and other time.b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2?
=type.googleapis.com/tensorflow.profiler.GenericRecommendation?
nohigh"t53.0 % of the total step time sampled is spent on 'All Others' time. This could be due to Python execution overhead.:
Refer to the TF2 Profiler FAQb?38.784% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 