?($	K?`A???R???r???Q???!?A`??"??$	zO]?<?@?'
?@?P?;m?@!?_?2q0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$ڬ?\m????C??????Aŏ1w-??Y?ʡE????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????????s??A???A(??y??Yw-!?l??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??? ?r??1?*????AW?/?'??Y?~j?t???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&j?q?????Gx$(??AyX?5?;??Y46<?R??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Fx$?? A?c?]??A1?Zd??YtF??_??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??9#J{??-!?lV??A??z6???Yu????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?z?G???n????A?V-??Y?3??7??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??Q????N@a???A??ׁsF??Y"??u????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?A`??"???D?????A?H?}??Y333333??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	b??4?8??H?z?G??A?߾?3??Y?'????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
??u????q???h??A?I+???Y333333??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&*??D????	h"lx??AK?46??YT㥛? ??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&=,Ԛ????????M??Aa??+e??Y????o??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&X9??v????Zd;???AV????_??Y?j+??ݓ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&u?V???[ A???A?(\?????Y?ݓ??Z??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&pΈ?????鷯????At??????Y8??d?`??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&B?f??j??9EGr???AQ?|a??Y?&S???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Zd;?O??M?O????AM?J???Y????Mb??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ڊ?e???8EGr???A??y???Y???{????"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&ı.n???-??????A6?>W[???YX9??v???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?Q???0L?
F%??A?/?$??Y????Mb??*	?????,?@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeats??A???!???@@)q=
ףp??1K< AV?@:Preprocessing2F
Iterator::Model鷯????!]?T'gvB@)R???Q??1PT?R`?5@:Preprocessing2U
Iterator::Model::ParallelMapV2??@?????!Ո???&.@)??@?????1Ո???&.@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::ZipZd;?O???!?s?ؘ?O@)E???JY??1Gmе$@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSlicef?c]?F??!??ȕ|Q!@)f?c]?F??1??ȕ|Q!@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::ConcatenateV}??b??!?f??p/@)????o??1"<?,>@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMap`vOj??!??Vp=3@)K?=?U??1????&@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*'???????!?J?(??@)'???????1?J?(??@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 47.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9L#8F? @>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	?5?A?}???;\??????C??????!n????	!       "	!       *	!       2$	????????E???{???/?$??!?H?}??:	!       B	!       J$	??kù??U*?αD??X9??v???!?ʡE????R	!       Z$	??kù??U*?αD??X9??v???!?ʡE????JCPU_ONLYYL#8F? @b Y      Y@q?"?6@@"?
both?Your program is POTENTIALLY input-bound because 47.9% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?32.426% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 