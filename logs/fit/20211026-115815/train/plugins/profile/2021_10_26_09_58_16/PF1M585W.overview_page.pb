?($	w????&??7d^??????T???N??!?4?8EG??$	?5m?&K@????p@?>?jEk@!yƔ???1@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$?T???N???4?8EG??Ac?=yX??Y??ܵ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&%??C?????ڊ?e??A?镲q??Y+??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&2U0*?????:M??ANё\?C??Ya??+e??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& ?o_???%u???AtF??_??Yr??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?:M???9EGr???A]?Fx??Y?q??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&t$???~????H?}??A?Ǻ????YZd;?O???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ǻ????? c?ZB>??A?Fx$??Y$????ۗ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????Mb???:M???A???????Y䃞ͪϕ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&P??n?????ǘ????As??A???Y??ܵ?|??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	?ׁsF???????_v??A??? ?r??YK?=?U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?Pk?w????q?????A ?~?:p??Y??d?`T??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?~j?t???	??g????A䃞ͪ???Y??_vO??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&B>?٬????!??u???A?ׁsF???Yy?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?????Q?|a??AP??n???Ysh??|???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&)??0???vOjM??A?3??7??Y???<,Ԫ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?4?8EG??V????_??A??	h"l??Y)\???(??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&W?/?'??t$???~??A7?[ A??Y'???????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????S??V-?????A?٬?\m??Y????<,??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&&䃞ͪ????ڊ?e??Affffff??Y?N@aÓ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????z???):????A?^)????Y???<,Ԛ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??0?*???}8gD??A?C??????Yŏ1w-!??*	333333?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?H?}8??!-----@A@)?<,Ԛ???1<<<<<@@:Preprocessing2F
Iterator::Model?i?q????!?????sB@)ꕲq???1??????8@:Preprocessing2U
Iterator::Model::ParallelMapV2?z?G???!yxxxx (@)?z?G???1yxxxx (@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???????!yxxxx?O@)??V?/???1?????L%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice~??k	???!     x@)~??k	???1     x@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate~8gDi??!?????1,@)?!??u???1LKKKK?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap<Nё\???!?1@)???x?&??1[ZZZZ?@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?0?*??!?@)?0?*??1?@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 46.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9 S1?T@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?@???????????"???4?8EG??!V????_??	!       "	!       *	!       2$	/???????i??Ƥ???c?=yX??!??	h"l??:	!       B	!       J$	)?A?????rm?|?j??K?=?U??!??ܵ???R	!       Z$	)?A?????rm?|?j??K?=?U??!??ܵ???JCPU_ONLYY S1?T@b Y      Y@q?T?i?I@"?
both?Your program is POTENTIALLY input-bound because 46.3% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?51.3001% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 