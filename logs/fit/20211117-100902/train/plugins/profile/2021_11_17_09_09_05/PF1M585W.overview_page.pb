?($	n,R??????ٿ?u????9#J{???! A?c?]??$	M۽c%@???4?@p?z@!?|??+0@"e
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails$v?????????(???A
h"lxz??Y?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?}8gD???v??/??AF%u???Y	?^)ˠ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails& A?c?]??p_?Q??A???H.??Y?N@aã?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&D????9???W[?????AjM??S??Y䃞ͪϥ?"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?鷯???|?5^???A??d?`T??Y? ?	???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&???ZӼ???uq???A?y?):???Y???~?:??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&????S???^)????A??(\????Ym???{???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??ܵ??Ϊ??V???A&䃞ͪ??Y???QI??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?ׁsF???ё\?C???A?/?'??Y?o_???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&	I.?!???????T????A[Ӽ???Y7?[ A??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&
;pΈ?????W?2??A?c]?F??Y?b?=y??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&6?>W[???ˡE?????A?,C????Y?H?}??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?W?2??䃞ͪ???AJ{?/L???Y????<,??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&Ș?????gDio????Ac?ZB>???Yp_?Q??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?V?/?'??Ǻ?????Av??????Y?&1???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&)\???(??v??????A???????Yd?]K???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&R'????????d?`T??A$???~???Y@?߾???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??%䃞??6<?R?!??A?ׁsF???Y?MbX9??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&??0?*???^)???AгY?????YvOjM??"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?u???????6?[??A?6?[ ??Y?!??u???"g
=type.googleapis.com/tensorflow.profiler.PerGenericStepDetails&?9#J{????Ǻ????A??V?/???Y2U0*???*	     ȓ@2l
5Iterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat?_vO??!Fb5\??B@)?#??????1x?7?M@A@:Preprocessing2F
Iterator::Model-??????!P???s@@)V-?????1?m9?.?3@:Preprocessing2U
Iterator::Model::ParallelMapV2?K7?A`??!*X??a*@)?K7?A`??1*X??a*@:Preprocessing2Z
#Iterator::Model::ParallelMapV2::Zip?v??/??!X???x?P@)?n?????1=q(|?%@:Preprocessing2?
OIterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate[0]::TensorSliceI??&??!??>M	@)I??&??1??>M	@:Preprocessing2v
?Iterator::Model::ParallelMapV2::Zip[0]::FlatMap[2]::Concatenate????!!???t+@)?ڊ?e???14?Syt?@:Preprocessing2f
/Iterator::Model::ParallelMapV2::Zip[0]::FlatMapڬ?\m???!J?E+?2@)??镲??1?,?_?
@:Preprocessing2x
AIterator::Model::ParallelMapV2::Zip[1]::ForeverRepeat::FromTensor*??镲??!?,?_?
@)??镲??1?,?_?
@:Preprocessing:?
]Enqueuing data: you may want to combine small input data chunks into fewer but larger chunks.
?Data preprocessing: you may increase num_parallel_calls in <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map" target="_blank">Dataset map()</a> or preprocess the data OFFLINE.
?Reading data from files in advance: you may tune parameters in the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch size</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave cycle_length</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer_size</a>)
?Reading data from files on demand: you should read data IN ADVANCE using the following tf.data API (<a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch" target="_blank">prefetch</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/Dataset#interleave" target="_blank">interleave</a>, <a href="https://www.tensorflow.org/api_docs/python/tf/data/TFRecordDataset#class_tfrecorddataset" target="_blank">reader buffer</a>)
?Other data reading or processing: you may consider using the <a href="https://www.tensorflow.org/programmers_guide/datasets" target="_blank">tf.data API</a> (if you are not using it now)?
:type.googleapis.com/tensorflow.profiler.BottleneckAnalysis?
both?Your program is POTENTIALLY input-bound because 52.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).no*no9H?<?@>Look at Section 3 for the breakdown of input time on the host.B?
@type.googleapis.com/tensorflow.profiler.GenericStepTimeBreakdown?
$	??ѧ??L??^q????Ǻ????!䃞ͪ???	!       "	!       *	!       2$	^?W6???=E?%?????V?/???!v??????:	!       B	!       J$	?*|?????e?|"???!??u???!@?߾???R	!       Z$	?*|?????e?|"???!??u???!@?߾???JCPU_ONLYYH?<?@b Y      Y@q &??R@"?
both?Your program is POTENTIALLY input-bound because 52.4% of the total step time sampled is spent on 'All Others' time (which could be due to I/O or Python execution or both).b
`input_pipeline_analyzer (especially Section 3 for the breakdown of input operations on the Host)m
ktrace_viewer (look at the activities on the timeline of each Host Thread near the bottom of the trace view)"T
Rtensorflow_stats (identify the time-consuming operations executed on the CPU_ONLY)"Z
Xtrace_viewer (look at the activities on the timeline of each CPU_ONLY in the trace view)*?
?<a href="https://www.tensorflow.org/guide/data_performance_analysis" target="_blank">Analyze tf.data performance with the TF Profiler</a>*y
w<a href="https://www.tensorflow.org/guide/data_performance" target="_blank">Better performance with the tf.data API</a>2I
=type.googleapis.com/tensorflow.profiler.GenericRecommendation
nono:
Refer to the TF2 Profiler FAQb?75.3597% of Op time on the host used eager execution. Performance could be improved with <a href="https://www.tensorflow.org/guide/function" target="_blank">tf.function.</a>2"CPU: B 