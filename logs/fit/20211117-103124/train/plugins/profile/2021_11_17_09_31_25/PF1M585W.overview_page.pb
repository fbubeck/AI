?($	?O?Ŗ???z[???0*??D??!?'????$	>?C\?@i??0?l@?v?[@!?????5@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$P?s???y?&1???A]m???{??Y-??臨?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?St$?????1??%???Aa2U0*???YjM????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????z???????AKY?8????Y?0?*??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&1?Zd????o_??AO??e???YjM????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?f??j+???i?q????A?	h"lx??YˡE?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?3??7????"??~j??Ai o????Y-C??6??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????8gDio???A?_?L??YHP?sג?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??s??????	h"l??A\ A?c???Y_?Qڛ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ڊ?e???J?4??A???_vO??Y?p=
ף??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	ףp=
???u?V??AbX9????Y??ʡE???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
?'?????[ A???AT㥛? ??Y#??~j???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???z6??(??y??Ay?&1???Y??y?):??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&}гY??????ܵ???A"lxz?,??Y)\???(??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&a??+e??#J{?/L??AGx$(??Y㥛? ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?1w-!??Q?|a??A????????Y%u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&-??????z?):????A?A?f???Yh??|?5??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??|?5^??c?ZB>???A??%䃞??Y?ݓ??Z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?S㥛???T㥛? ??A?T???N??Y"??u????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&aTR'????}?5^?I??A(??y??Y	?^)ː?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&S?!?uq???????K??AV????_??Y??~j?t??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&0*??D???MbX9??Ad]?Fx??Y??&???*	gffff??@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeatV????_??!d?vf@@@)?ڊ?e???1ô#0>@:Preprocessing2F
Iterator::Model???3???!??eN??B@)?_vO??1?gޓ7@:Preprocessing2U
Iterator::Model::ParallelMapV2?;Nё\??!??}??*@)?;Nё\??1??}??*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?q??????!ic???O@)?	h"lx??1??x7?i'@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice?:pΈҮ?!??r??B@)?:pΈҮ?1??r??B@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate?	?c??!??Yaa?*@)?ʡE????1??@@?}@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap䃞ͪ???!3,?>?J3@)?	h"lx??1??x7?i@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*?I+???!Ym?)??@)?I+???1Ym?)??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 51.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9??}o?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?]K?=????s?Q????MbX9??!?[ A???	!       "	!       *	!       2$	y??H?
???f%?/A??d]?Fx??!T㥛? ??:	!       B	!       J$	????1]????/t???	?^)ː?!-??臨?R	!       Z$	????1]????/t???	?^)ː?!-??臨?JCPU_ONLYY??}o?@b Y      Y@q[_?ӉU@"?
both?Your program is POTENTIALLY input-bound because 51.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?84.399% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 