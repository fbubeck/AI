?($	2;a[?????=???????MbX??!X9??v???$	???AH@?~??4h@?%??????!???^|)@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$??????gDio????A#J{?/L??Y{?G?z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?	h"lx?? ?o_???A^K?=???Y?b?=y??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&㥛? ????ZB>????A????z??Y?0?*???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ǘ????J{?/L???A?HP???Y=?U?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&d?]K???M?J???A????_v??Y???3???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&X9??v???F%u???A?}8gD??Y?(??0??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?lV}?????????A???????Yw-!?l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?q?????>?٬?\??Aio???T??Y??0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&3ı.n????C?l????A??N@a??Yu????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?R?!?u?????镲??A?镲q??Y??Pk?w??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
???<,????G?z??AW[??????Y??ׁsF??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ǻ????M?O???A?߾?3??Y?&S???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&J{?/L?????V?/???Aŏ1w-!??Y?<,Ԛ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????=yX?5??A]?Fx??Y??y?):??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?G?z????Zd;??A????Y%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&EGr????^K?=???A2U0*???Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&46<?R??Z??ڊ???Aq???h??Y?j+??ݓ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&P??n????:pΈ???A??1??%??YHP?s??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ۊ?e????-??????A??y?):??Y?z6?>??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&2w-!????MbX9??A?=?U???Y??ܵ?|??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??MbX?????h o??A??j+????Y???QI??*	fffffZ?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat??+e???!??7?s?A@)؁sF????1o6?W/@@:Preprocessing2F
Iterator::ModelӼ????!?{h??B@)?sF????1?E??8@:Preprocessing2U
Iterator::Model::ParallelMapV2J+???!#?b	N:*@)J+???1#?b	N:*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip{?/L?
??!?f??	hO@)#??~j???1Lzn??%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::Concatenate[0]::TensorSlicez6?>W??!R{??8?@)z6?>W??1R{??8?@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[0]::ConcatenateNbX9???!?	????)@)?J?4??1E??@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap??????!F??&a?0@)??Ɯ?1????@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*w-!?l??!0?0?q@)w-!?l??10?0?q@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 53.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9Z?n?N@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?=????s0T?????h o??!F%u???	!       "	!       *	!       2$	??H.?!???.?~?????j+????!io???T??:	!       B	!       J$	w?,????
ygX??????QI??!??0?*??R	!       Z$	w?,????
ygX??????QI??!??0?*??JCPU_ONLYYZ?n?N@b Y      Y@q?E?=?kU@"?
both?Your program is POTENTIALLY input-bound because 53.0% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?85.6838% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 