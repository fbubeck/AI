?($	?؞p?q???w?????Ӽ????!??V?/???$	"
-/?@??I???@ԇi]%B@!??8r0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$???QI???_?L??A+??ݓ???Yı.n???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?J?4???ݓ??Z??A?9#J{???Yu????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??"??~??~8gDi??A??????Y$????ۗ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?|гY??????Mb??A??z6???Y?Q?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????_v??|a2U0??A?^)????YK?=?U??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&'?W??????B?i??A$(~??k??Y?#??????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????tF??_??A??{??P??YX9??v???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??V?/???{?/L?
??AaTR'????Y;?O??n??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&U0*????U???N@??A?N@a???Y??ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	???h o?????(???A????_v??Y??&???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??~j?t??[B>?٬??A???(???Y8??d?`??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???????????????AX9??v??Y??j+????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Q???????K7??A????B???YQ?|a2??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?h o???	??g????A?~j?t???Y=?U?????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&#??~j??????9#J??A_)?Ǻ??Y?z6?>??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?9#J{????W?2??A/?$???Y/n????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?A`??"?????(???AHP?s???Y???Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?uq???o?ŏ1??AU???N@??Y??ZӼ???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?	h"lx?????ׁs??AbX9????Yn????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&=?U??????ׁsF??A???h o??Y?g??s???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ӽ?????U???د?A?ݓ??Z??Y?N@aÓ?*	???????@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat???H.??!Y??*?@@)V}??b??1??3???>@:Preprocessing2F
Iterator::Model?????!?L???@)??:M???1rlM3@:Preprocessing2U
Iterator::Model::ParallelMapV2?"??~j??!7??5S)@)?"??~j??17??5S)@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip???o_??!?,:>?Q@)?Fx$??1ѣtʡR%@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate??	h"??!?a???1@)sh??|???1SrZ???"@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlice??&S??!?Pp?? @)??&S??1?Pp?? @:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapW?/?'??!ؚ???7@)?\m?????1:?P?>w@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*	?c???!+??[??@)	?c???1+??[??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 57.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9?????@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	t?@???Z?~?u%???U???د?!tF??_??	!       "	!       *	!       2$	?p?`/'??I?/????ݓ??Z??!????_v??:	!       B	!       J$	]ɼ?????sEGSM???Q?????!ı.n???R	!       Z$	]ɼ?????sEGSM???Q?????!ı.n???JCPU_ONLYY?????@b Y      Y@q?\H?B@"?
both?Your program is POTENTIALLY input-bound because 57.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?37.4006% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 